
import torch.optim
from models.AMSF import AMSF
from utils import *
from validate_houston import validate
from train_CAVE import train
import pdb
import args_parser
from build_datasets_houston import *


args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)


def main():

    train_ref, train_hr, test_ref, test_lr, test_hr = all_train_test_data_in()
    if args.dataset == 'CAVE':
      args.n_bands = 31
      args.n_select_bands = 3
    elif args.dataset == 'houston':
      args.n_bands = 46
      args.n_select_bands = 3
    elif args.dataset == 'chikusei':
      args.n_bands = 110
      args.n_select_bands = 8

    # Build the models

    model = AMSF(args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands
                 ).cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=500,
                                                gamma=0.95)
    parameter_nums = sum(p.numel() for p in model.parameters())
    print("Model size:", str(float(parameter_nums / 1e6)) + 'M')
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(test_ref, test_lr, test_hr,args.image_size,
                         model,
                          0,
                          args.n_epochs)
        print ('psnr: ', recent_psnr)

    # # Loss and Optimizer
    criterion = nn.MSELoss().cuda()
    #
    #
    best_psnr = 0
    best_psnr = validate(test_ref, test_lr, test_hr,args.image_size,
                         model,
                          0,
                          args.n_epochs)
    print ('psnr: ', best_psnr)

    # Epochs
    print ('Start Training: ')
    best_epoch = 0
    total_loss = []
    step = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    for epoch in range(args.n_epochs):
        step = step + 1
        loss = train(train_ref,
              train_hr,
              args.image_size,
              args.batch_size,
              model,
              optimizer,
              scheduler,
              criterion,
        )
        loss = loss.cpu()
        loss = loss.detach()
        loss.np = loss.numpy()
        total_loss.append(loss.np)


        # One epoch's validation
        recent_psnr = validate(test_ref, test_lr, test_hr,args.image_size,
                         model,
                          epoch,
                          args.n_epochs)
        if epoch % 1==0:
            print('Train_Epoch_{}: '.format(epoch))
            print('psnr: ', recent_psnr)
        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)

        if is_best:
          best_epoch=epoch
          if best_psnr > 0:
            torch.save(model.state_dict(), model_path)
            print('Saved!     best psnr:', best_psnr, 'at epoch:', best_epoch)
            model.load_state_dict(torch.load(model_path), strict=False)
            print('')

    print('best_psnr: ', best_psnr)



if __name__ == '__main__':
    main()
