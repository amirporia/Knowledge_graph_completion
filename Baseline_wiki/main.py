import torch
import json
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import datetime

from Baseline.setting.config import args
from Baseline.model.trainer import Trainer
from Baseline.setting.logger_config import logger


def main():
    cudnn.benchmark = True

    if args.distributed:
        # Launched via `torchrun --nproc_per_node=N ...`: args.rank/local_rank/world_size
        # were populated from RANK/LOCAL_RANK/WORLD_SIZE in config.setup_distributed().
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.dist_backend, init_method='env://',
            device_id=torch.device(f'cuda:{args.local_rank}'),
            timeout=datetime.timedelta(minutes=60),
        )
        logger.info(
            f'Distributed training: rank {args.rank}/{args.world_size} '
            f'(local GPU {args.local_rank}, backend={args.dist_backend})'
        )
    else:
        ngpus = torch.cuda.device_count()
        logger.info(f'Use {ngpus} gpu(s) for training' if ngpus else 'No GPU available, using CPU')

    trainer = Trainer(args, ngpus_per_node=torch.cuda.device_count())

    # Only rank 0 needs to log the full args dump; every rank would otherwise print an
    # identical block.
    if args.rank == 0:
        logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))

    trainer.train_loop()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()