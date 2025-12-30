import argparse

def opt_algorithm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'wide',help='indicator to dataset')
    # path setting
    parser.add_argument('--result_path', type=str, default= 'result/',help='path to the folder to save results')
    
    # experiment controls
    parser.add_argument('--modality', type=str, default= 'v',help='choose modality for experiment: v, s, v+s')
    parser.add_argument('--posthoc_la', action='store_true', help='Posthoc LA for state update')
    parser.add_argument('--mode', type=str, default= 'train',help='select from train, val, test. Used in dataset creation')
    parser.add_argument('--net_v', type=str, default= 'resnet18',help='choose network backbone for image channel: vgg19bn, resnet18, resnet50, wrn, wiser') 
    parser.add_argument('--net_s', type=str, default= 'gru',help='choose network backbone for ingredient channel: gru, nn')
    parser.add_argument('--method', type=str, default= 'align',help='choose method backbone in: align, clip, img2word')    
    parser.add_argument('--dim_latent', type=int, default = 1024, help='dim of latent in gru')
    parser.add_argument('--vit_drop', type=float, default = 0., help='dim of latent in gru')
    # loss weight
    parser.add_argument('--num_test', default=10, type=int, help='Curriculum Test')
    parser.add_argument('--accept_rate', type=float, default=0.6, help='Increasing accept ratio')
    parser.add_argument('--pos_weight', type=float, default = 40, help='positive weight of bcelosswithlogits in nus-wide dataset')
    # alignment
    parser.add_argument('--pht_partial', type=float, default= 0.3,help='coefficient of cmfl')
    # model_pretrain
    parser.add_argument('--encoder_finetune', action='store_true', help='finetune visual and semantic encoder')
    
    # rebalancing methods
    parser.add_argument('--rebalance', type=str, default= 'none',help='focal, ldam_drw, cb_resample, cb_reweight') 
    parser.add_argument('--rebalances', type=str, default= 'none',help='focal, ldam_drw, cb_resample, cb_reweight') 
    parser.add_argument('--gamma_focal', type=float, default= 0.,help='gamma of focal loss')
    parser.add_argument('--beta_rw', type=float, default= 0.9999,help='beta of class-balanced reweight')
    parser.add_argument('--beta_rs', type=float, default= 0.9999,help='beta of class-balanced resample')
    parser.add_argument('--m_ldam', type=float, default= 1,help='m of ldam')
    parser.add_argument('--s_ldam', type=float, default= 1,help='scale of ldam')
    parser.add_argument('--epoch_drw', type=int, default= 8,help='epoch to use reweight')
    # cross-modal methods
    parser.add_argument('--beta_cmfl', type=float, default= 0.,help='coefficient of cmfl')
    
    parser.add_argument('--beta_vae_kl', type=float, default= 1.,help='beta of kl')
    parser.add_argument('--beta_vae', type=float, default= 1.,help='beta of vae')
    parser.add_argument('--beta_cm_vae', type=float, default= 1.,help='beta of vae')  
    parser.add_argument('--lr_cls', type=float, default= 1.,help='beta of vae')
    parser.add_argument('--method_disentangle', type=str, default= '',help='beta of vae')
    parser.add_argument('--beta_vae_cross_dd', type=float, default= 1.,help='beta of vae')
    
    # turning parameters
    
    parser.add_argument('--batch_size', type=int, default = 64, help='batch size')
    parser.add_argument('--lr', type=float, default = 5e-4, help='learning rate')
    parser.add_argument('--lr_finetune', type=float, default = 1e-5, help='fine-tune learning rate')
    parser.add_argument('--lrd_rate', type=float, default = 0.1, help='decay rate of learning rate')
    parser.add_argument('--lrd_rate_finetune', type=float, default = 0.1, help='decay rate of fine-tune learning rate')
    parser.add_argument('--lr_decay', type=int, default = 4, help='decay rate of learning rate')
    parser.add_argument('--weight_decay', type=float, default = 1e-3, help='weight decay')

    parser.add_argument('--w_align', type=float, default = 0.1, help='coefficient of  l2norm loss between image features and word features')
    parser.add_argument('--w_align2', type=float, default = 0.1, help='coefficient of  l2norm loss between image features and word features')
    parser.add_argument('--type_align', type=str, default = 'l2', help='type of align loss: kl or l2')
    parser.add_argument('--w_semantic', type=float, default = 1.0, help='word prediction loss')
    # cmi methods
    parser.add_argument('--topk', type=int, default = 20,help='top num for img2word prediction')
    parser.add_argument('--cmi', type=str, default= 'linear',help='methods of cmi')
    parser.add_argument('--beta_pri', type=float, default = 0.1,help='soft factor for class adj, from 0.0~1.0, the larger, the softer')
    parser.add_argument('--beta_loss_cls', type=float, default = 1.0,help='loss weight of class-aware loss')
    parser.add_argument('--adj', type=str, default= 'ho',help='recon_soft_sum')
    parser.add_argument('--beta_class', type=int, default= 1,help='weight of gcn class')
    parser.add_argument('--gcn_threshold', type=float, default=0.5,help='the bigger, the more different from exist tage realations, range from [0,0.5]')
    parser.add_argument('--dim_embed', type=int, default = 256, help='the latent dim of embedding')

    #TDE
    parser.add_argument('--alpha', type=float, default = 0.4,help='alpha of TDE')
    parser.add_argument('--num_head', type=int, default=2, help='num_head of TDE')
    parser.add_argument('--tau', type=int, default =16 , help='tau of TDE')
    parser.add_argument('--gamma', type=float, default =0.003125,help='gamma of TDE')
    parser.add_argument('--remethod', type=str, default ='xERM',help='gamma of TDE')
    # feature fusion
    parser.add_argument('--beta_fusion', type=float, default = 0.1,help='beta of feature fusion in add method')
    parser.add_argument('--method_fusion', type=str, default= 'add',help='methods of feature fusion')
    parser.add_argument('--pretrain_model', type=int, default =9,help='top num for img2word prediction')
    #cifar
    parser.add_argument('--cifar_imb_ratio', type=float, default =0.01,help='ratio imbalance of cifar')
    parser.add_argument('--beta_cmo', type=float, default =0.7,help='ratio  of cmo')
    parser.add_argument('--ext_str', type=str, default = '', help='extra string')
    parser.add_argument('--alpha_a', type=float, default = 0.4,help='alpha of Gpaco')
    parser.add_argument('--moco-dim', default=768, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=32, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')    
    args = parser.parse_args()
    
    return args