import dataset.image_datasets

def from_name(name, data_folder, batch_size, output_size):
    if name == dataset.image_datasets.CelebA.name:
        return dataset.image_datasets.CelebA(data_folder=data_folder, batch_size=batch_size, output_shapes=[output_size])
    elif name == dataset.image_datasets.ImageNet.name:
        return dataset.image_datasets.ImageNet(data_folder=data_folder, batch_size=batch_size, output_shapes=[output_size])
    elif name == dataset.image_datasets.LSunBedroom.name:
        return dataset.image_datasets.LSunBedroom(data_folder=data_folder, batch_size=batch_size, output_shapes=[output_size], center_crop=(256, 256))
    else:
        raise ValueError('Unknown dataset')