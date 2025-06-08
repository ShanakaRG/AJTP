import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# from plot_motion_intensity import plot_graph2
from torch.fft import fft , rfft 
import matplotlib.pyplot as plt 
import torch.nn.functional as F

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class AdaSampling_frame_based_energy_selective_joint(nn.Module):
    def __init__(self,scaling_factor,layer,selective_joint):
        super (AdaSampling_frame_based_energy_selective_joint,self).__init__()
        # self.sampling = nn.Linear(int(input_T/sampling_parameter), input_t) 
        self.n_norm = 0.5
        self.n_rect = 5
        self.epsilon = 0.0001
        self.epsilon_2 = 0.0000001
        self.scaling_factor =scaling_factor
        self.layer = layer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.no_joints = 8
        self.selective_joint=selective_joint

    def forward(self,x):
        x =x.permute(0,1,3,2)
        N ,C, V , T = x.shape
        L1 = (1/x.shape[1])*torch.sum(torch.abs(((x[:,:,:,1:T] - x[:,:,:,0:T-1]))),1).cuda()

        if self.selective_joint ==1:
            L1_energy = (1/x.shape[1])*torch.sum(torch.abs(((x[:,:,:,1:T] - x[:,:,:,0:T-1]))),1).cuda() #calculate the energy of the joints 
            energy = torch.sum(torch.square(L1_energy) , 2)
            poten = (1/x.shape[1])*torch.sum(((x[:,:,:,0:T] - x[:,:,:,0].unsqueeze(-1))),1).cuda()
            poten_energy = torch.abs(torch.sum(poten , 2))
            max_elements_pot, max_idxs_pot = torch.max(poten_energy ,1)
            energy_norm_pot = torch.div(poten_energy ,max_elements_pot.unsqueeze(-1))
            mean_energy_pot = torch.mean(energy_norm_pot,1)
            max_elements, max_idxs = torch.max(energy,1)
            energy_norm = torch.div(energy ,max_elements.unsqueeze(-1))

            max_elements_total, max_idxs_total = torch.max(energy_norm + energy_norm_pot ,1)
            energy_norm_total = torch.div((energy_norm + energy_norm_pot),max_elements_total.unsqueeze(-1))

            mean_energy = torch.mean(energy_norm,1)
            
            mean_energy_total = torch.mean(energy_norm_total ,1)
            std_energy_total = torch.std(energy_norm_total,1)

            active_joint_mask = ((energy_norm_total - mean_energy_total[:,None])>=-0.01*std_energy_total[:, None] )
            
            active_joints  =  L1*active_joint_mask.unsqueeze(-1).repeat(1, 1, L1.shape[2])
            L_active = torch.div(torch.sum(active_joints,1) ,active_joint_mask.sum(axis=1).unsqueeze(-1)) .cuda()
            
            if torch.any(L_active .isnan()):
                    raise ValueError("L_active  contains nan values") 
            elif torch.any(L_active .isinf()):
                    raise ValueError("L_active  contains inf values")
            m = nn.Tanh()
            I_t = m(L_active )
        else: 

            L1 = (1/x.shape[2])*torch.sum(L1,1) 
            m = nn.Tanh()
            I_t = m(self.n_norm*L1 )
        I_t = I_t/torch.sum(I_t,1).reshape(-1,1)
        I_t_norm = torch.zeros((x.shape[0],T))
        I_t_norm[:,1:] = I_t
        P_t = torch.cumsum(I_t_norm, axis=1)
        tau = int(T/self.scaling_factor)
        w = (1/tau)*(1+(1/(T-1)))
        
        B_i= [-1/(2*(T-1)) + w*i for i in range(0, tau+1)]
        m_i= [0.5*(B_i[i]+B_i[i+1]) for i in range (0 ,tau)]
        m_i = torch.tensor(m_i)
        ###########################for plots ###################################################
        
        # fig, axs = plt.subplots(nrows=2, ncols=2, dpi=1000)
        # fig.suptitle('plots for Action ')
        # ax_big = fig.add_subplot(2, 2, (2,4))

        # print(energy_norm_total.shape)
        # axs[0, 0].stem(energy_norm_total.detach().cpu().numpy()[0])
        # axs[0, 0].axhline(y=mean_energy_total[0], color='r', linestyle='--')
        # # ax_big.plot(P_t0.detach().numpy().T,label = 'original activeJ')
        # print(P_t.shape)
        # # fd
        # ax_big.plot(P_t[0].detach().numpy().T,label = 'tanh_selectiveJ')
        # # ax_big.plot(P_t2.detach().numpy().T,label = 'tanh_selectiveJ')
        # # ax_big.plot(P_t4.detach().numpy().T,label = 'new_exp')
        # # ax_big.plot(P_t3.detach().numpy().T,label = 'old_exp')
        # # ax_big.plot(P_t5.detach().numpy().T,label = '100tanh10_poten')
        # # ax_big.plot(P_t5_.detach().numpy().T,label = '100tanh10_poten+energy')
        # ax_big.plot(L1[0].detach().cpu().numpy().T,label = 'L1_active')
        # # ax_big.plot(L_poten.detach().numpy().T,label = 'poten')

        # x1 = [0, T]                    # Define the x values as a list of two points
        # y1 = [0, 1]
        # ax_big.plot(x1, y1,  '-r', linestyle='--',linewidth=1)
        # for xc in B_i:
        #     ax_big.axhline(y=xc, color='gray', linestyle='--')
        # # plt.plot(P_t.T)

        # # ax_big.set_yticklabels([i, for i in range (0,1)])
        # # ax_big.set_xticklabels([])
        # axs[0, 1].remove()
        # axs[1, 1].remove()
        # # ax_big = axs[0, 1].merge(axs[1, 0])
        # ax_big.set_xlim([0, T])
        # ax_big.set_ylim([0,1])
        # # ax_big.legend(loc='upper center',prop={'size': 8}, handlelength=0.5, handleheight=0.5)

        # axs[1, 0].axis('off')
        # leg = axs[1, 0].legend(handles=ax_big.get_lines(),
        #             #    labels=['original activeJ','tanh-allJ','tanh_selectiveJ','new_exp','old_exp','L1_active',  'poten'],
        #             #    labels=['exp','sigmoid','tanh','relu','MI'],
        #                labels=['selectJ','MI'],
        #                loc='center')



        # plt.savefig('/data/srg079/code/AdaPool/CTR-GCN/data/ntu120/figs/testing_data_figures/frame_joint_selection_plots_for_Action__layer'+str(self.layer)+ '_.png')
        
        ####################################################################################
        M = torch.zeros(P_t.shape[0],tau,T)
        for N in range (0,P_t.shape[0]):
            
            M[N,:,:] = (1 / (torch.pow(((torch.sub(P_t[N,:].reshape(1,-1),m_i.reshape(-1,1))) / (0.5*w) ),(2*self.n_rect) ) +1 ) ).cuda()
            
            
            if torch.any(M.isnan()):
                raise ValueError("M  contains nan values")  
            elif torch.any(M.isinf()):
                raise ValueError("M  contains inf values") 
            
        M_norm = (M/(torch.sum(M, 2).unsqueeze(-1) + self.epsilon )).to(self.device)
        x =x.permute(0,3,1,2)
        x_temp = torch.reshape(x,(x.shape[0],T, x.shape[2]*x.shape[3])).float()
        out = (torch.matmul( M_norm,x_temp)).cuda()
        out = torch.reshape(out,(x.shape[0],tau, x.shape[2],x.shape[3]))
        out =out.permute(0,2,1,3)
        return out ,out


class AdaSampling_joint_wise_selective_energy(nn.Module):
    def __init__(self,scaling_factor,layer,selective_joint):
        super (AdaSampling_joint_wise_selective_energy,self).__init__()
        # self.sampling = nn.Linear(int(input_T/sampling_parameter), input_t) 
        self.n_norm = 0.5
        self.n_rect = 5
        self.epsilon = 0.0001
        self.epsilon_2 = 0.0000001
        self.scaling_factor =scaling_factor
        self.layer = layer
        self.no_joints = 8
        self.selective_joint = selective_joint
        

    def forward(self,x):
        '''
        input shape = [ number of samples , channels, frames,  , joints]
        '''
        x =x.permute(0,1,3,2)
        N ,C, V , T = x.shape
        L1 =  (1/x.shape[1])*torch.sum(torch.abs(((x[:,:,:,1:T] - x[:,:,:,0:T-1]))),1).cuda() # sum over only channels 
        
        if self.selective_joint == 1:
            L1_energy = (1/x.shape[1])*torch.sum(torch.abs(((x[:,:,:,1:T] - x[:,:,:,0:T-1]))),1) #calculate the energy of the joints 
            energy = torch.sum(torch.square(L1_energy) , 2)
            poten = (1/x.shape[1])*torch.sum(((x[:,:,:,0:T] - x[:,:,:,0].unsqueeze(-1))),1).cuda()
            poten_energy = torch.abs(torch.sum(poten , 2))
            max_elements_pot, max_idxs_pot = torch.max(poten_energy ,1)
            energy_norm_pot = torch.div(poten_energy ,max_elements_pot.unsqueeze(-1))
            mean_energy_pot = torch.mean(energy_norm_pot,1)
    
            max_elements, max_idxs = torch.max(energy,1)
            
            energy_norm = torch.div(energy ,max_elements.unsqueeze(-1))

            max_elements_total, max_idxs_total = torch.max(energy_norm + energy_norm_pot ,1)
            energy_norm_total = torch.div((energy_norm + energy_norm_pot),max_elements_total.unsqueeze(-1))

            mean_energy = torch.mean(energy_norm,1)
            
            mean_energy_total = torch.mean(energy_norm_total ,1)
            std_energy_total = torch.std(energy_norm_total,1)

            active_joint_mask = ((energy_norm_total - mean_energy_total[:,None])>=-0.25*std_energy_total[:, None] )
            L_active =  L1*active_joint_mask.unsqueeze(-1).repeat(1, 1, L1.shape[2])
            m = nn.Tanh()
            I_t = m(L_active)
        else: 
            m = nn.Tanh()
            I_t = m(L1)
        
        I_t = I_t / (torch.sum(I_t,2).unsqueeze(-1)+ 1e-8)
        I_t_norm = torch.zeros((N,V,T))
        I_t_norm[:,:, 1:] = I_t
        P_t = torch.cumsum(I_t_norm, axis=2)
        tau = int(T/self.scaling_factor)
        w = (1/tau)*(1+(1/(T-1)))

        B_i= [-1/(2*(T-1)) + w*i for i in range(0, tau+1)]
        m_i= [0.5*(B_i[i]+B_i[i+1]) for i in range (0 ,tau)]
        m_i = torch.tensor(m_i)
        M = torch.zeros(N,V, tau,T)
        ###############################for plots###############################################
        # print(energy_norm_total.shape)
        # axs[0, 0].stem(energy_norm_total.detach().numpy()[0])
        # axs[0, 0].axhline(y=mean_energy_total[0], color='r', linestyle='--')
        # fig, axs = plt.subplots(nrows=2, ncols=2, dpi=1000)
        # fig.suptitle('plots for Action ')
        # ax_big = fig.add_subplot(2, 2, (2,4))
        # # ax_big.plot(P_t0.detach().numpy().T,label = 'original activeJ')
        # print(P_t.shape)
        # # fd
        # ax_big.plot(P_t[0,:,:].detach().numpy().T,label = 'tanh_selectiveJ')
        # # ax_big.plot(P_t2.detach().numpy().T,label = 'tanh_selectiveJ')
        # # ax_big.plot(P_t4.detach().numpy().T,label = 'new_exp')
        # # ax_big.plot(P_t3.detach().numpy().T,label = 'old_exp')
        # # ax_big.plot(P_t5.detach().numpy().T,label = '100tanh10_poten')
        # # ax_big.plot(P_t5_.detach().numpy().T,label = '100tanh10_poten+energy')
        # ax_big.plot(L1[0,:,:].detach().cpu().numpy().T,label = 'L1_active')
        # # ax_big.plot(L_poten.detach().numpy().T,label = 'poten')

        # x1 = [0, 64]                    # Define the x values as a list of two points
        # y1 = [0, 1]
        # ax_big.plot(x1, y1,  '-r', linestyle='--',linewidth=1)
        # for xc in B_i:
        #     ax_big.axhline(y=xc, color='gray', linestyle='--')
        # # plt.plot(P_t.T)

        # # ax_big.set_yticklabels([i, for i in range (0,1)])
        # # ax_big.set_xticklabels([])
        # axs[0, 1].remove()
        # axs[1, 1].remove()
        # # ax_big = axs[0, 1].merge(axs[1, 0])
        # ax_big.set_xlim([0, 64])
        # ax_big.set_ylim([0,1])
        # # ax_big.legend(loc='upper center',prop={'size': 8}, handlelength=0.5, handleheight=0.5)

        # axs[1, 0].axis('off')
        # leg = axs[1, 0].legend(handles=ax_big.get_lines(),
        #             #    labels=['original activeJ','tanh-allJ','tanh_selectiveJ','new_exp','old_exp','L1_active',  'poten'],
        #             #    labels=['exp','sigmoid','tanh','relu','MI'],
        #                labels=['selectJ','MI'],
        #                loc='center')



        # plt.savefig('/data/srg079/code/AdaPool/CTR-GCN/data/ntu120/figs/testing_data_figures/frame_joint_selection_plots_for_Action_ .png')
    
        ####################################################################################
        M = (1 / (torch.pow(((torch.sub(P_t.reshape(1,-1),m_i.reshape(-1,1)).view(N,V, tau,T).permute(1,2,0,3)) / (0.5*w) ),(2*self.n_rect) ) +1 ) ).cuda()                    
        M_norm = (M/(torch.sum(M, 3).unsqueeze(-1) + self.epsilon )) .cuda()
        x =x.permute(0,2,3,1)
        x = (torch.matmul( M_norm.permute(2,0,1,3),x)).cuda()
        x= x.permute(0,3,2,1)
        x= torch.where(torch.isnan(x), torch.tensor(0.).cuda(), x)
        if torch.any(x.isnan()):
            raise ValueError("X  contains nan values") 
        elif torch.any(x.isinf()):
            raise ValueError("X  contains inf values") 
        
        return x , x

class SubClassifer(nn.Module):
    def __init__(self, num_joints, feature_dim, num_classes,frames,kernal_size=3):
        super(SubClassifer, self).__init__()
        self.fc1 = nn.Linear( num_joints*feature_dim,num_joints)
        self.fc2 = nn.Linear( num_joints,num_classes)
        self.pool1d = nn.AdaptiveMaxPool1d(1)
        self.drop_out = nn.Dropout(p=0.3)
        
    def mask_gen (self, softmax_output, threshold=0.85):
      sorted_indices = torch.argsort(softmax_output, dim=1, descending=True)
      
      # Calculate the cumulative sum
      cumulative_sum = torch.cumsum(softmax_output[torch.arange(softmax_output.size(0)).unsqueeze(1), sorted_indices], dim=1)
      mask = torch.where(cumulative_sum > threshold, torch.tensor(1).cuda(), torch.tensor(-1).cuda())
      masked_sort_index = (sorted_indices+1) * mask.squeeze(-1)
      masked_sort_index  = torch.clamp(masked_sort_index , min=0)
      joint_mask = torch.zeros(masked_sort_index.shape)
      joint_mask[torch.arange(masked_sort_index.shape[0]).unsqueeze(1), masked_sort_index-1] = 1
      joint_mask = joint_mask.long()
      return joint_mask
    
    def forward (self,x):
      N,C,T,V = x.shape

      x= x.permute(0,1,3,2).contiguous ()
      x= x.view(N,C*V,T)
      x = self.pool1d(x).squeeze(-1)
      x= x.view(N,C,V)
      x= x.view(N,C*V)
      x = self.fc1(x)
      mask = F.softmax(x,dim=1)
      mask = self.mask_gen(mask).cuda()
      x=self.fc2(x)
      x = self.drop_out(x)

      
      return x ,mask
    


class AdaSampling_frame_based_learnable_selective_joint(nn.Module):
    def __init__(self,scaling_factor,layer,selective_joint,num_joints, feature_dim,  num_classes,frames):
        super (AdaSampling_frame_based_learnable_selective_joint,self).__init__()
        # self.sampling = nn.Linear(int(input_T/sampling_parameter), input_t) 
        self.n_norm = 0.5
        self.n_rect = 5
        self.epsilon = 0.0001
        self.epsilon_2 = 0.0000001
        self.scaling_factor =scaling_factor
        self.layer = layer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.no_joints = 8
        self.selective_joint=selective_joint
        self.mask= SubClassifer(num_joints, feature_dim,  num_classes,int(frames), kernal_size=3)

    def forward(self,x):
        x =x.permute(0,1,3,2)
        N ,C, V , T = x.shape
        L1 = (1/x.shape[1])*torch.sum(torch.abs(((x[:,:,:,1:T] - x[:,:,:,0:T-1]))),1).cuda()

        if self.selective_joint ==1:
            predicts_out,active_joint_mask = self.mask(x.permute(0,1,3,2).contiguous ())
            active_joints  =  L1*active_joint_mask .unsqueeze(-1).repeat(1, 1, L1.shape[2])
            L_active = torch.div(torch.sum(active_joints,1) ,active_joint_mask .sum(axis=1).unsqueeze(-1)) .cuda()
            if torch.any(L_active .isnan()):
                    raise ValueError("L_active  contains nan values") 
            elif torch.any(L_active .isinf()):
                    raise ValueError("L_active  contains nan values") 
            m = nn.Tanh()
            I_t = m(L_active )
           
        else: 
            L1 = (1/x.shape[2])*torch.sum(L1,1) 
            m = nn.Tanh()
            I_t = m(self.n_norm*L1 )
        I_t = I_t/torch.sum(I_t,1).reshape(-1,1)
        I_t_norm = torch.zeros((x.shape[0],T))
        I_t_norm[:,1:] = I_t
        P_t = torch.cumsum(I_t_norm, axis=1)
        tau = int(T/self.scaling_factor)
        w = (1/tau)*(1+(1/(T-1)))
        
        B_i= [-1/(2*(T-1)) + w*i for i in range(0, tau+1)]
        m_i= [0.5*(B_i[i]+B_i[i+1]) for i in range (0 ,tau)]
        m_i = torch.tensor(m_i)
        ##########################for plots####################################################
        '''
        fig, axs = plt.subplots(nrows=2, ncols=2, dpi=1000)
        fig.suptitle('plots for Action ')
        ax_big = fig.add_subplot(2, 2, (2,4))

        print(energy_norm_total.shape)
        axs[0, 0].stem(energy_norm_total.detach().cpu().numpy()[0])
        axs[0, 0].axhline(y=mean_energy_total[0], color='r', linestyle='--')
        # ax_big.plot(P_t0.detach().numpy().T,label = 'original activeJ')
        print(P_t.shape)
        # fd
        ax_big.plot(P_t[0].detach().numpy().T,label = 'tanh_selectiveJ')
        # ax_big.plot(P_t2.detach().numpy().T,label = 'tanh_selectiveJ')
        # ax_big.plot(P_t4.detach().numpy().T,label = 'new_exp')
        # ax_big.plot(P_t3.detach().numpy().T,label = 'old_exp')
        # ax_big.plot(P_t5.detach().numpy().T,label = '100tanh10_poten')
        # ax_big.plot(P_t5_.detach().numpy().T,label = '100tanh10_poten+energy')
        ax_big.plot(L1[0].detach().cpu().numpy().T,label = 'L1_active')
        # ax_big.plot(L_poten.detach().numpy().T,label = 'poten')

        x1 = [0, T]                    # Define the x values as a list of two points
        y1 = [0, 1]
        ax_big.plot(x1, y1,  '-r', linestyle='--',linewidth=1)
        for xc in B_i:
            ax_big.axhline(y=xc, color='gray', linestyle='--')
        # plt.plot(P_t.T)

        # ax_big.set_yticklabels([i, for i in range (0,1)])
        # ax_big.set_xticklabels([])
        axs[0, 1].remove()
        axs[1, 1].remove()
        # ax_big = axs[0, 1].merge(axs[1, 0])
        ax_big.set_xlim([0, T])
        ax_big.set_ylim([0,1])
        # ax_big.legend(loc='upper center',prop={'size': 8}, handlelength=0.5, handleheight=0.5)

        axs[1, 0].axis('off')
        leg = axs[1, 0].legend(handles=ax_big.get_lines(),
                    #    labels=['original activeJ','tanh-allJ','tanh_selectiveJ','new_exp','old_exp','L1_active',  'poten'],
                    #    labels=['exp','sigmoid','tanh','relu','MI'],
                       labels=['selectJ','MI'],
                       loc='center')



        plt.savefig('/data/srg079/code/AdaPool/CTR-GCN/data/ntu120/figs/testing_data_figures/frame_joint_selection_plots_for_Action_'+str(self.layer)+ '_.png')
        '''



        ####################################################################################
        M = torch.zeros(P_t.shape[0],tau,T)
        for N in range (0,P_t.shape[0]):
            M[N,:,:] = (1 / (torch.pow(((torch.sub(P_t[N,:].reshape(1,-1),m_i.reshape(-1,1))) / (0.5*w) ),(2*self.n_rect) ) +1 ) ).cuda()
            if torch.any(M.isnan()):
                raise ValueError("M  contains nan values")  
            elif torch.any(M.isinf()):
                raise ValueError("M contains nan values") 
        M_norm = (M/(torch.sum(M, 2).unsqueeze(-1) + self.epsilon )).to(self.device)
        x =x.permute(0,3,1,2)
        x_temp = torch.reshape(x,(x.shape[0],T, x.shape[2]*x.shape[3])).float().to(self.device)
        out = (torch.matmul( M_norm,x_temp)).cuda()
        out = torch.reshape(out,(x.shape[0],tau, x.shape[2],x.shape[3]))
        out =out.permute(0,2,1,3)
        return out , predicts_out

class AdaSampling_joint_based_learnable_selective_joint(nn.Module):
    def __init__(self,scaling_factor,layer,selective_joint,num_joints, feature_dim,  num_classes,frames):
        super (AdaSampling_joint_based_learnable_selective_joint,self).__init__()
        self.n_norm = 0.5
        self.n_rect = 5
        self.epsilon = 0.0001
        self.epsilon_2 = 0.0000001
        self.scaling_factor =scaling_factor
        self.layer = layer
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.selective_joint=selective_joint
        self.mask= SubClassifer(num_joints, feature_dim,  num_classes,int(frames), kernal_size=3)


    def forward(self,x):
        x =x.permute(0,1,3,2)
        N ,C, V , T = x.shape
        
        L1 =  (1/x.shape[1])*torch.sum(torch.abs(((x[:,:,:,1:T] - x[:,:,:,0:T-1]))),1).cuda() # sum over only channels 
        
        if self.selective_joint == 1:
            predicts_out,active_joint_mask = self.mask(x.permute(0,1,3,2).contiguous ())
            L_active   =  L1*active_joint_mask .unsqueeze(-1).repeat(1, 1, L1.shape[2])
            swapped_active_joint_mask = torch.where(active_joint_mask .unsqueeze(-1).repeat(1, 1, L1.shape[2]) == 0, torch.tensor(1).cuda(), torch.tensor(0).cuda())
            replacement_vector = torch.arange(1, T) .unsqueeze(0).unsqueeze(0).cuda()
            uniform_motion = swapped_active_joint_mask * replacement_vector
            L_active = L_active+uniform_motion
            m = nn.Tanh()
            I_t = m(L_active)
        else: 
            m = nn.Tanh()
            I_t = m(L1)
        I_t = I_t / (torch.sum(I_t,2).unsqueeze(-1)+ 1e-8)
        I_t_norm = torch.zeros((N,V,T))
        I_t_norm[:,:, 1:] = I_t
        P_t = torch.cumsum(I_t_norm, axis=2)
        tau = int(math.ceil(T/self.scaling_factor))
        w = (1/tau)*(1+(1/(T-1)))

        B_i= [-1/(2*(T-1)) + w*i for i in range(0, tau+1)]
        m_i= [0.5*(B_i[i]+B_i[i+1]) for i in range (0 ,tau)]
        m_i = torch.tensor(m_i)
        M = torch.zeros(N,V, tau,T)
        M = (1 / (torch.pow(((torch.sub(P_t.reshape(1,-1),m_i.reshape(-1,1)).view(N,V, tau,T).permute(1,2,0,3)) / (0.5*w) ),(2*self.n_rect) ) +1 ) ).cuda()                  
        M_norm = (M/(torch.sum(M, 3).unsqueeze(-1) + self.epsilon )) .cuda()
        if torch.any(M_norm.isnan()):
            raise ValueError("M  contains nan values") 
        elif torch.any(M_norm.isinf()):
            raise ValueError("M contains nan values") 
        x =x.permute(0,2,3,1)
        if torch.any(x.isnan()):
            raise ValueError("X contains nan values") 
        elif torch.any(x.isinf()):
            raise ValueError("X contains nan values") 
        x = (torch.matmul( M_norm.permute(2,0,1,3),x)).cuda()
        out= x.permute(0,3,2,1)
        if torch.any(out.isnan()):
            raise ValueError("Out  contains nan values") 
        elif torch.any(out.isinf()):
            raise ValueError("Out contains nan values") 
        return out , predicts_out

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

class TCN_Pool_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, layer,selective_joint, scaling_factor =2, res_stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2], num_joints=25, num_classes=120,frames=64,pooling_strategy ='joint',learnable_pooling =True):
        super(TCN_Pool_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        if pooling_strategy == 'joint':
            if learnable_pooling: 
                self.adapool =AdaSampling_joint_based_learnable_selective_joint(scaling_factor,layer,selective_joint,num_joints, out_channels,  num_classes,frames)
            else: 
                self.adapool = AdaSampling_joint_wise_selective_energy(scaling_factor,layer,selective_joint)
               
        elif pooling_strategy == 'frame':
            if learnable_pooling:
                self.adapool =AdaSampling_frame_based_learnable_selective_joint(scaling_factor,layer,selective_joint,num_joints, out_channels,  num_classes,frames)
            else:
                self.adapool = AdaSampling_frame_based_energy_selective_joint(scaling_factor,layer,selective_joint)
        else: 
            raise ValueError ('undefined pooling strategy')
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=1, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (res_stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=res_stride)

    def forward(self, x):
        x1 ,predicts_out= self.adapool(self.gcn1(x))
        
        
        return self.relu(self.tcn1(x1)+ self.residual(x)) , predicts_out


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, selective_joint=True,pooling_strategy = 'joint',learnable_pooling=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.selective_joint=selective_joint
        self.pooling_strategy = pooling_strategy
        self.learnable_pooling = learnable_pooling
        
        base_channel = 64
        base_frames = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_Pool_GCN_unit(base_channel, base_channel*2, A,5, selective_joint= self.selective_joint,scaling_factor =2 , res_stride=2,  adaptive=adaptive,  num_joints = self.num_point, num_classes=self.num_class, frames = base_frames,pooling_strategy =self.pooling_strategy,learnable_pooling=self.learnable_pooling )
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_Pool_GCN_unit(base_channel*2, base_channel*4, A, 8,selective_joint= self.selective_joint,scaling_factor =2 , res_stride=2,  adaptive=adaptive,  num_joints = self.num_point, num_classes=self.num_class, frames = base_frames/2,pooling_strategy=self.pooling_strategy, learnable_pooling=self.learnable_pooling)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x) # N * M, C, T, V
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x , predict_l5= self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x , predict_l8= self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        predict_l5 = predict_l5.view(N,M,-1).mean(1)
        predict_l8 = predict_l8.view(N,M,-1).mean(1)
        
        return self.fc(x) , predict_l5,  predict_l8