

from models.AMSF import AMSF

from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_cc, calc_moae, calc_uiqi, calc_ssim

import args_parser
# from torch.nn import functional as F
# import cv2
from time import *
# from thop import profile
args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from build_datasets_houston import *
print(args)

def main():

    args.n_bands = 46

    _, _, test_ref, test_lr, test_hr = all_train_test_data_in()
    # Build the models

    model = AMSF(
                     args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands).cuda()


    # Load the trained model parameters
    model_path = "F:\stt\code\MCT-Net-main\ASMF/houston_arch.pkl"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print('Load the chekpoint of {}'.format(model_path))


    # test_ref, test_lr, test_hr = test_list
    test_ref = torch.Tensor(test_ref)
    test_lr = torch.Tensor(test_lr)
    test_hr = torch.Tensor(test_hr)

    test_ref = test_ref.permute(0, 3, 1, 2)
    test_lr = test_lr.permute(0, 3, 1, 2)
    test_hr = test_hr.permute(0, 3, 1, 2)

    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()
    model.cuda()
    lr = lr.cuda()
    hr = hr.cuda()

    out,_,_,_ = model(lr, hr)




    # flops
    print()
    print()

    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    refs = np.split(ref, 7, axis=0)
    ref = [np.squeeze(a, axis=0) for a in refs]
    outs = np.split(out, 7, axis=0)
    out = [np.squeeze(a, axis=0) for a in outs]

    rmse = [0]*len(ref)
    psnr = [0]*len(ref)
    PSNR = AverageMeter()
    for i in range(len(ref)):
        sio.savemat('./result/{}_{}_ref.mat'.format(i, args.dataset), {'ref':np.squeeze(refs[i], 0).transpose(1, 2, 0)})
        sio.savemat('./result/{}_{}_out.mat'.format(i, args.dataset), {'out':np.squeeze(outs[i], 0).transpose(1, 2, 0)})   #保存字典
    print('{:.4f}', PSNR.avg)




if __name__ == '__main__':
    main()
