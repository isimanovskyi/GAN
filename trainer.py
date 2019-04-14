import numpy as np
import torch
import optimizers

class Trainer(object):
    def __init__(self, model, batch, loss, lr, reg, lambd):
        self.model = model
        self.batch = batch
        self.sub_batches = 1

        self.loss = loss
        self.reg = reg
        self.lambd = lambd
        self._init_optimizer(lr)

    def _get_r1_reg(self, real_d, img):
        grads = torch.autograd.grad(real_d.sum(), img, retain_graph=True, create_graph=True, only_inputs=True)[0]
        batch_size = img.size(0)
        reg = grads.pow(2).view(batch_size, -1).sum(1).mean()
        return reg

    def _get_r2_reg(self, fake_d, g_samples):
        grads = torch.autograd.grad(fake_d.sum(), g_samples, retain_graph=True, create_graph=True, only_inputs=True)[0]
        batch_size = g_samples.size(0)
        reg = grads.pow(2).view(batch_size, -1).sum(1).mean()
        return reg

    def _get_gp_reg(self, img, gen_samples, norm = 'l2'):
        epsilon = torch.rand(img.shape[0], device=self.model.device).reshape((img.shape[0],1,1,1))

        t = img + epsilon * (gen_samples - img)
        t.detach_()
        t.requires_grad = True

        d = self.model.d_model(t)
        grads = torch.autograd.grad(d.sum(), t, retain_graph=True, create_graph=True, only_inputs=True)[0]

        if norm == 'l2':
            reg = grads.view(grads.shape[0], -1).pow(2).sum(1).mean()
        elif norm == 'l1':
            reg = grads.view(grads.shape[0], -1).abs().sum(1).pow(2).mean()
        else:
            raise ValueError('Unknown norm')

        del t
        del epsilon
        del grads
        del d

        return reg

    def _get_reg(self, img, gen_samples):
        if self.reg == 'gp':
            return self._get_gp_reg(img, gen_samples)
        else:
            return None

    def _init_optimizer(self, lr):
        d_vars, g_vars = self.model.get_weights()

        #self.d_optim = torch.optim.SGD(d_vars, lr)
        #self.g_optim = torch.optim.SGD(g_vars, lr)
        self.d_optim = torch.optim.RMSprop(d_vars, lr)
        #self.g_optim = torch.optim.RMSprop(g_vars, lr)
        self.g_optim = optimizers.TRRmsProp(self._g_opt_loss, torch.optim.RMSprop(g_vars, lr/2, weight_decay=1e-3), delta=0.5, verbose=True)
        #self.d_optim = optimizers.RMSpropEx(d_vars, lr)
        #self.g_optim = optimizers.RMSpropEx(g_vars, lr/3.)

    def process_d_batch(self):
        b = self.batch.get()
        img = b['images'].to(self.model.device)

        real_d = self.model.d_model(img)
        d_real_loss = self.loss.get_d_real(real_d)
        d_real_loss.backward()

        with torch.no_grad():
            z = b['z'].to(self.model.device)
            gen_samples = self.model.g_model(z)

        #fake
        fake_d = self.model.d_model(gen_samples)
        d_fake_loss = self.loss.get_d_fake(fake_d)
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss

        #regularizer
        reg = self._get_reg(img, gen_samples)
        if reg is not None:
            d_reg = float(self.lambd) * reg
            d_reg.backward()
        return d_loss, reg

    def update_d(self, n_steps):
        self.model.d_model.requires_grad(True)
        self.model.g_model.requires_grad(False)

        err_D = 0.
        err_S = 0.
        for _ in range(n_steps):
            #print ('update_d')
            self.d_optim.zero_grad()

            #update gradient
            m = float(self.sub_batches)
            for i in range(self.sub_batches):
                d_loss, d_reg = self.process_d_batch()
                d_loss /= m
                d_reg /= m

                err_D += d_loss.data.cpu().numpy()
                err_S += np.sqrt(d_reg.data.cpu().numpy())

            self.d_optim.step()

        M = float(n_steps)
        err_D /= M
        err_S /= M
        return err_D, err_S

    def process_g_batch(self):
        z = self.batch.get_z().to(self.model.device)
        gen_samples = self.model.g_model(z)

        fake_d = self.model.d_model(gen_samples)
        g_loss = self.loss.get_g(fake_d)
        g_loss.backward()

        if hasattr(self.g_optim, 'z_lst'):
            self.g_optim.z_lst.append((z, gen_samples))
        return g_loss

    def _g_opt_loss(self, z_list):
        r = 0.
        for z, samples in z_list:
            gen_samples = self.model.g_model(z)
            r += (samples - gen_samples).abs().mean()

        r /= float(len(z_list))
        r = r.view(1, ).data.cpu().numpy()[0]
        return r

    def update_g(self, n_steps):
        self.model.d_model.requires_grad(False)
        self.model.g_model.requires_grad(True)

        err_G = 0.
        for _ in range(n_steps):
            #zero_grad
            self.g_optim.zero_grad()

            #update gradient#
            m = float(self.sub_batches)
            for i in range(self.sub_batches):
                g_loss = self.process_g_batch()
                g_loss /= m

                err_G += g_loss.data.cpu().numpy()

            self.g_optim.step()

        self.model.update_g_av()

        err_G /= float(n_steps)
        return err_G

    def update(self, d_steps, g_steps):
        self.model.train()

        #train D
        b_too_good_D = False
        for _ in range(100):
            errD, s = self.update_d(d_steps)
            if errD < self.loss.zero_level:
                break
            b_too_good_D = True
        if errD > self.loss.zero_level:
            raise RuntimeError('Discriminator cannot distinguish')

        #train G
        errG = self.update_g(g_steps)

        return errD, s, errG, b_too_good_D

    def sample(self, z):
        self.model.d_model.requires_grad(False)
        self.model.g_model.requires_grad(False)

        z = torch.tensor(z, device = self.model.device)

        with torch.no_grad():
            if hasattr(self.model, 'av_g_model'):
                self.model.av_g_model.eval()
                gen_samples = self.model.av_g_model(z)
            else:
                self.model.g_model.eval()
                gen_samples = self.model.g_model(z)
                self.model.g_model.train()

            gen_samples = torch.clamp(gen_samples, -1., 1.)
            res = gen_samples.data.cpu().numpy()

        return res
