from framework.engine.tester import Tester


if __name__ == "__main__":
    test = Tester('/home/fedor/datasets/NewYorkDataset/example.png', '/home/fedor/projects/checkpoint_epoch_91.pth',
                  '/home/fedor/projects/log.txt', '/home/fedor/datasets/NewYorkDataset')

    with open('/home/fedor/projects/test_list.txt', 'r') as f:
        file_list = f.read().split('\n')

    for indx, sample in enumerate(file_list):
        path = '/home/fedor/projects/ScannTech/frames/' + sample
        if indx == 0:
            test.update_model(path)
        print(path)
        test.test_sample(path, '/home/fedor/projects/ScannTech/vidoe/' + f'{indx}.jpg')
