import matplotlib.pyplot as plt
import numpy as np

savepath = r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/concat/'

#kitti2015
# x = np.load(r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/kitti2015_bsconv_500_x.npy')

# knpk2015_y = np.load(r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/kitti2015_original_500_y.npy')
# knpdw3k2015_y = np.load(r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/kitti2015_bsconv_500_y.npy')

# knpk2015noc_y = np.load(r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/kitti2015_original_500_noc_y.npy')
# knpdw3k2015noc_y = np.load(r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/kitti2015_bsconv_500_noc_y.npy')

# plt.plot(x, knpk2015_y, label = 'baseline(occ)')
# plt.plot(x, knpdw3k2015_y, label = 'proposed(occ)')
# plt.plot(x, knpk2015noc_y, label = 'baseline(noc)')
# plt.plot(x, knpdw3k2015noc_y, label = 'proposed(noc)')
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('3-pixel error')
# plt.title('KITTI 2015')
# plt.savefig(savepath + 'kitti2015' + '.png')


#kitti2012
x = np.load(r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/kitti2012_bsconv_300_x.npy')

knpk2012_y = np.load(r'/home/youngmin/YM/PSMNet-master/test_results/kitti-no-pretrained-kitti2012_y.npy')
knpdw3k2012_y = np.load(r'/home/youngmin/YM/PSMNet-master/test_results/kitti-no-pretrained-dw-3-kitti2012_y.npy')
bsconv = np.load(r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/kitti2012_bsconv_300_y.npy')

knpk2012noc_y = np.load(r'/home/youngmin/YM/PSMNet-master/test_results/kitti-no-pretrained-kitti2012-noc_y.npy')
knpdw3k2012noc_y = np.load(r'/home/youngmin/YM/PSMNet-master/test_results/kitti-no-pretrained-dw-3-kitti2012-noc_y.npy')
bsconv_noc = np.load(r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/kitti2012_bsconv_300_noc_y.npy')


plt.plot(x, knpk2012_y, label = 'baseline(occ)')
plt.plot(x, knpdw3k2012_y, label = 'PSMNet+dwconv(occ)')
plt.plot(x, bsconv, label = 'Proposed(occ)')

plt.plot(x, knpk2012noc_y, label = 'baseline(noc)')
plt.plot(x, knpdw3k2012noc_y, label = 'PSMNet+dwconv(noc)')
plt.plot(x, bsconv_noc, label = 'Proposed(noc)')

plt.legend()
plt.xlabel('epochs')
plt.ylabel('3-pixel error')
plt.title('KITTI 2012')
plt.savefig(savepath + 'kitti2012_300' + '.png')


#sceneflow
# x2 = np.load(r'/home/youngmin/YM/PSMNet-master/test_results/sf-no-pretrained-dw3_x.npy')

# sfnp_y = np.load(r'/home/youngmin/YM/PSMNet-master/test_results/sf-no-pretrained_y.npy')
# sfnpdw3_y = np.load(r'/home/youngmin/YM/PSMNet-master/test_results/sf-no-pretrained-dw3_y.npy')

# plt.plot(x2, sfnp_y, label = 'baseline')
# plt.plot(x2, sfnpdw3_y, label = 'proposed')
# plt.legend()
# plt.xlabel('checkpoints')
# plt.ylabel('test loss')
# plt.title('Scene Flow')
# plt.savefig(savepath + 'SceneFlow' + '.png')
