import dataset.image_datasets

def from_name(name, batch_size, output_size):
    if name == dataset.image_datasets.CelebA.name:
        return dataset.image_datasets.CelebA(batch_size = batch_size, output_shapes = [output_size])
    elif name == dataset.image_datasets.ImageNet.name:
        return dataset.image_datasets.ImageNet(batch_size = batch_size, output_shapes = [output_size])
    elif name == dataset.image_datasets.LSunBedroom.name:
        return dataset.image_datasets.LSunBedroom(batch_size = batch_size, output_shapes = [output_size], center_crop = (256, 256))
    else:
        raise ValueError('Unknown dataset')