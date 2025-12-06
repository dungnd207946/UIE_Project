import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config.yml',
                        help='YAML file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], default='val',
                        help='Use validation split for evaluation')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None,
                        help='GPU ids, e.g. "0" or "0,1"; leave empty for CPU only')
    parser.add_argument('-debug', '-d', action='store_true',
                        help='Debug mode: smaller dataset, more frequent logging')
    parser.add_argument('-enable_wandb', action='store_true',
                        help='Enable Weights & Biases logging (optional)')
    args = parser.parse_args()

    # Parse config & setup logging
    opt = Logger.dict_to_nonedict(Logger.parse(args))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'eval', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info('Evaluation config:\n' + Logger.dict2str(opt))

    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Dataset (dùng đúng UIEDataset trong data/dataset.py)
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished ({} samples)'.format(len(val_set)))

    # Model (DDPM + UNet + GaussianDiffusion)
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

    # Evaluation loop
    logger.info('Begin Evaluation...')
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    sum_psnr = 0.0
    sum_ssim = 0.0
    n_img = 0

    for idx, val_data in enumerate(val_loader, start=1):
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()

        restore_img = Metrics.tensor2img(visuals['output'])   # uint8, [H,W,3]
        target_img = Metrics.tensor2img(visuals['target'])    # uint8, [H,W,3]

        psnr = Metrics.calculate_psnr(restore_img, target_img)
        ssim = Metrics.calculate_ssim(restore_img, target_img)

        sum_psnr += psnr
        sum_ssim += ssim
        n_img += 1

        logger.info('Image {:4d} | PSNR: {:.4f} dB | SSIM: {:.4f}'.format(idx, psnr, ssim))
        tb_logger.add_scalar('eval/psnr', psnr, idx)
        tb_logger.add_scalar('eval/ssim', ssim, idx)

    if n_img == 0:
        logger.warning('No images found in validation loader.')
        return

    avg_psnr = sum_psnr / n_img
    avg_ssim = sum_ssim / n_img

    logger.info('# Evaluation # Avg PSNR: {:.4f} dB'.format(avg_psnr))
    logger.info('# Evaluation # Avg SSIM: {:.4f}'.format(avg_ssim))

    print('====== EVALUATION FINISHED ======')
    print('Average PSNR: {:.4f} dB'.format(avg_psnr))
    print('Average SSIM: {:.4f}'.format(avg_ssim))


if __name__ == '__main__':
    main()
