import os
import torch
import utils
import logger
import shutil

class Checkpoint(object):
    def __init__(self, checkpoint_dir, file_prefix = 'chk'):
        self.modules = {}
        self.checkpoint_dir = checkpoint_dir;
        self.file_prefix = file_prefix
        utils.exists_or_mkdir(checkpoint_dir)

    def register(self, name, obj, mandatory):
        if name in self.modules.keys():
            raise ValueError('Module already registered')

        self.modules[name] = (obj, mandatory)

    def save(self, it):
        logger.info("[*] Saving checkpoint ...")

        if it > 0:
            filename = self.file_prefix + '_' + str(it) + '.npz'
            latest_filename = self.file_prefix + '.npz'
        else:
            filename = self.file_prefix + '.npz'
            latest_filename = None

        filename = os.path.join(self.checkpoint_dir, filename)
        
        outdict = {'it':it}
        for n, (o, m) in utils.get_items(self.modules):
            outdict[n] = o.state_dict()
        
        torch.save(outdict, filename)

        if latest_filename is not None:
            latest_filename = os.path.join(self.checkpoint_dir, latest_filename)
            shutil.copyfile(filename, latest_filename)

        logger.info("[*] Saving checkpoint SUCCESS!")

    def load(self, it):
        logger.info("[*] Loading checkpoint ...")
        if it > 0:
            filename = self.file_prefix + '_' + str(it) + '.npz'
        else:
            filename = self.file_prefix + '.npz'

        filename = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filename):
            logger.info('[*] checkpoint not found')
            return 0

        outdict = torch.load(filename)
        it = outdict['it']
        for n, (o, m) in utils.get_items(self.modules):
            if n in outdict:
                o.load_state_dict(outdict[n])
            else:
                if m:
                    raise RuntimeError('Cannot load mandatory object')
                logger.info("Cannot load object: \'" + n + "\'")

        logger.info('[*] checkpoint loaded')
        return it