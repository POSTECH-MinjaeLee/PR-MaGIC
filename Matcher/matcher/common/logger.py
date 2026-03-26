r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = [ int(i.split('-')[0]+'00'+i.split('-')[1]) for i in dataset.class_ids] if dataset.benchmark=='dis' else dataset.class_ids  
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000
        elif self.benchmark == 'paco_part':
            self.nclass = 448
        elif self.benchmark == 'pascal_part':
            self.nclass = 100
        elif self.benchmark == 'lvis':
            self.nclass = 1203
        elif self.benchmark == 'dis':
            self.nclass = 210

        self.total_miou_buf = 0
        self.miou_buf = 0
        self.count = 0
        self.ones = 0
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        iou = (inter_b.float()/union_b.float())
        self.total_miou_buf += iou.mean()
        self.miou_buf = iou.mean()
        self.count +=1
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        # iou = self.intersection_buf.float() / \
        #       torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        # iou = iou.index_select(1, self.class_ids_interest)
        miou = self.total_miou_buf/self.count * 100

        # #fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
        #           self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100
        
        #return miou, fb_iou, iou[1]
        return miou, self.miou_buf

    def write_result(self, split, epoch):
        miou, iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % miou
        Logger.info(msg)
    
    def write_csv(self, split, epoch):
        iou = self.compute_iou()
        msg = '%5.2f,' % iou 

        Logger.info(msg)


    def write_process(self, batch_idx, datalen, epoch, nested, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else '**[nested: %d]'% nested
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            miou, iou= self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  | ' % miou
            msg += 'current IoU: %5.2f   ' % iou
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, root='logs'):
        logtime = datetime.datetime.now().__format__('%m%d_%H%M%S')
        logpath = '_TEST_' + logtime

        cls.logpath = os.path.join(root, logpath + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with Matcher ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

