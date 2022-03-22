import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.Sakuya_arch_o as Sakuya_arch_o
import models.modules.STVSR as STVSR
import models.modules.Sakuya_arch_test as Sakuya_arch_test
import models.modules.Sakuya_arch_test_continuous as Sakuya_arch_test_continuous
import models.modules.Sakuya_arch_test_single as Sakuya_arch_test_single
import models.modules.Sakuya_arch_test_nomul as Sakuya_arch_test_nomul
import models.modules.Sakuya_arch_test_noflow as Sakuya_arch_test_noflow
import models.modules.Sakuya_arch_test2 as Sakuya_arch_test2
import models.modules.Sakuya_arch_test3 as Sakuya_arch_test3
import models.modules.Sakuya_arch_test5 as Sakuya_arch_test5
import models.modules.Sakuya_arch_test_S as Sakuya_arch_test_S
####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'LIIF':
        netG = Sakuya_arch_test.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])   
    elif which_model == 'LunaTokis':
        netG = Sakuya_arch_o.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])  
    elif which_model == 'TMNet':
        netG = STVSR.TMNet(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])
    elif which_model == 'LIIF_test1':
        netG = Sakuya_arch_test.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])   
    elif which_model == 'LIIF_test2':
        netG = Sakuya_arch_test2.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])                           
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
