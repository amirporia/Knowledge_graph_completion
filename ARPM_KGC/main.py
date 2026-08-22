import torch
import json
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from ARPM_KGC.setting.config import args
from ARPM_KGC.model.trainer import Trainer
from ARPM_KGC.setting.logger_config import logger
from datetime import timedelta


def main():
    cudnn.benchmark = True

    if args.distributed:
        # Launched via `torchrun --nproc_per_node=N ...`: args.rank/local_rank/world_size
        # were populated from RANK/LOCAL_RANK/WORLD_SIZE in config.setup_distributed().
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.dist_backend, init_method='env://',
            device_id=torch.device(f'cuda:{args.local_rank}'),
            timeout=timedelta(minutes=60),
        )
        logger.info(
            f'Distributed training: rank {args.rank}/{args.world_size} '
            f'(local GPU {args.local_rank}, backend={args.dist_backend})'
        )
    else:
        ngpus = torch.cuda.device_count()
        logger.info(f'Use {ngpus} gpu(s) for training' if ngpus else 'No GPU available, using CPU')

    trainer = Trainer(args, ngpus_per_node=torch.cuda.device_count())

    if args.rank == 0:
        logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))

    trainer.train_loop()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
