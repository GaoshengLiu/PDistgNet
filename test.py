import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from utils import *
from model_PDistg import Net

from tqdm import tqdm
import scipy.io as sio
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--testset_dir', type=str, default='/root/data1/LF-SSR-methods/LF_fix/Data/TestData_4xSR_5x5/')

    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=32, help="The stride between two test patches is set to patchsize/2")

    parser.add_argument('--model_path', type=str, default='./log/PDistg_4xSR_5x5.pth.tar')#/root/data1/LF-SSR-methods/LF_fix
    parser.add_argument('--save_path', type=str, default='./Results/')

    return parser.parse_args()

def test(cfg, test_Names, test_loaders):

    #net = EDSR(cfg.angRes, cfg.upscale_factor)
    net = Net(cfg.angRes, cfg.upscale_factor)
    #net = net2x()
    net.to(cfg.device)
    # cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:1': cfg.device})
        net.load_state_dict(model['state_dict'])
    else:
        print("=> no model found at '{}'".format(cfg.model_path))
        # pass

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        ssim_testset_1 = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = inference(test_loader, test_name, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            #ssim_testset_1.append(ssim_epoch_test_1)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            txtfile = open(cfg.save_path + '/Metrics.txt', 'a')
            txtfile.write('Dataset----%15s,\t PSNR---%f,\t SSIM---%f\n' % (test_name, psnr_epoch_test, ssim_epoch_test))
            txtfile.close()
            pass
        pass


def inference(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    ssim_iter_test_1 = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angRes, vw // cfg.angRes
        subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angin, w*angin
        numU, numV, H, W = subLFin.shape
        #subLFout = torch.zeros(numU, numV, cfg.angout * cfg.patchsize*cfg.upscale_factor, cfg.angout * cfg.patchsize*cfg.upscale_factor)
        s = time.time()
        minibatch = 4
        num_inference = numU*numV//minibatch
        #print(num_inference)
        tmp_in = subLFin.contiguous().view(numU*numV, subLFin.shape[2], subLFin.shape[3])
        
        with torch.no_grad():
            out_lf = []
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:].unsqueeze(1)
                out_lf.append(net(tmp.to(cfg.device)))#
            if (numU*numV)%minibatch and numU*numV>minibatch:
                tmp = tmp_in[(idx_inference+1)*minibatch:,:,:].unsqueeze(1)
                out_lf.append(net(tmp.to(cfg.device)))#
        infer_time = time.time()-s
        # print(infer_time)
        out_lf = torch.cat(out_lf, 0)
        subLFout = out_lf.view(numU, numV, cfg.angRes * cfg.patchsize*cfg.upscale_factor, cfg.angRes * cfg.patchsize*cfg.upscale_factor)
        outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor, cfg.stride * cfg.upscale_factor, h0 * cfg.upscale_factor, w0 * cfg.upscale_factor)

        psnr, ssim = cal_metrics(label, outLF, cfg.angRes)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        #ssim_iter_test_1.append(ssim_1)

        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name)

        sio.savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                        {'LF': outLF.numpy()})
        pass


    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    #ssim_iter_test_1 = float(np.array(ssim_iter_test_1).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test#, ssim_iter_test_1


def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
