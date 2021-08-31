import torchvision
import torch

PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
PIXEL_STD = [1., 1., 1.]

# PIXEL_MEAN = [0.485, 0.456, 0.406]
# PIXEL_STD = [0.229, 0.224, 0.225]

def load_data(image_dir, batch_size, output_size=256, train=True):
    if train:
        transform_list = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(output_size),
                                                 torchvision.transforms.RandomHorizontalFlip(),
                                                 torchvision.transforms.ToTensor(),
                                                 lambda x: x*255,
                                                 torchvision.transforms.Normalize(mean=PIXEL_MEAN,
                                                                      std=PIXEL_STD),
                                                ])
    else:
        transform_list = torchvision.transforms.Compose([torchvision.transforms.Resize(output_size),
                                                 torchvision.transforms.CenterCrop(output_size),
                                                 torchvision.transforms.ToTensor(),
                                                 lambda x: x*255,
                                                 torchvision.transforms.Normalize(mean=PIXEL_MEAN,
                                                                      std=PIXEL_STD),
                                                ])
    
    data = torchvision.datasets.ImageFolder(root=image_dir, transform=transform_list)
    
    data_loader = torch.utils.data.DataLoader(data, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              drop_last=False, 
                                              num_workers=8)
    return data_loader