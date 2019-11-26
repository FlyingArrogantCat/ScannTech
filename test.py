from framework.engine.tester import Tester
import time


tester = Tester(ex_img_path='/home/fedor/Downloads/Telegram Desktop/23572.jpg',
                model_weigths_path='/home/fedor/Downloads/checkpoint_epoch_91.pth',
                log_path='smth', dataset_path=None)



img_path = '/home/fedor/Downloads/Telegram Desktop/20180108161434_9916.jpg'
tester.update_model(img_path)

t = time.clock()

tester.test_sample(img_path, img_save_path='/home/fedor/Desktop/res.jpeg')

print(f'Time: {time.clock() - t}')