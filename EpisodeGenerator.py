import typing
import os
import random
import numpy as np
from PIL import Image
import pickle

from utils import list_dir, list_files

def _load_image(img_url: str, expand_dim: bool = False) -> np.ndarray:
    """Load an image
    """
    img = Image.open(fp=img_url, mode='r')
    img_np = np.asarray(a=img, dtype=np.uint8)

    if len(img_np.shape) < 3: # gray scale image
        # insert channel into the array
        img_np = np.expand_dims(img_np, axis=0)

        if expand_dim:
            img_np = np.repeat(a=img_np, repeats=3, axis=0)
    else:
        img_np = np.transpose(a=img_np, axes=(2, 0, 1)) # convert to (c, h, w)
        img_np = img_np / 255 # normalize to [0, 1]

    return img_np

def _sample_uniform(num_samples: int, low: float, high: float) -> np.ndarray:
    return (low - high) * np.random.random_sample(size=num_samples) + high

def get_cls_img(root: str, suffix: str) -> dict:
    """Get folders from root, and images in each folder

    Args:
        root (str): the desired directory
        suffix (str): the suffix of file or image within each folder

    Returns: dictionary with keys are the folder names,
        and values are the lists of files within the corresponding folder
    """
    cls_img = dict.fromkeys(list_dir(root=root))
    for dir_ in cls_img:
        cls_img[dir_] = list_files(root=os.path.join(root, dir_), suffix=suffix)

    return cls_img

class OmniglotLoader(object):
    folder = '../datasets/omniglot-py/'

    def __init__(
        self,
        root: str = folder,
        train_subset: bool = True,
        suffix: str = '.png',
        min_num_cls: int = 5,
        max_num_cls: int = 20,
        k_shot: int = 20,
        expand_dim: bool = False,
        load_images: bool = True
    ) -> None:
        """Initialize a data loader for Omniglot data set or a two-level dataset
            with structure similar to Omniglot: alphabet -> character -> image

        Args:
            root (str): path to the folder of Omniglot data set
            train_subset (bool): if True, this will load data from
                the ``images_background`` folder (or, training set). If False,
                it loads data from ``images_evaluation``` (or, validation set)
            suffix (str): the suffix of images
            min_num_cls (int): minimum number of classes within a generated episode
            max_num_cls (int): maximum number of classes within a generated episode
            expand_dim (bool): if True, repeat the channel dimension from 1 to 3
            load_images (bool): if True, this will place all image data (PIL) on RAM.
                This option is optimal for small data set since it would speed up
                the data loading process. If False, it will load images whenever called.
                This option is suitable for large data set.

        Returns: an OmniglotLoader instance
        """
        self.root = os.path.join(root, 'images_background' if train_subset else 'images_evaluation')
        self.suffix = suffix
        self.min_num_cls = min_num_cls
        self.max_num_cls = max_num_cls
        self.k_shot = k_shot
        self.expand_dim = expand_dim
        self.load_images = load_images

        # create a nested dictionary to store data
        self.data = dict.fromkeys(list_dir(root=self.root))
        for alphabet in self.data:
            self.data[alphabet] = dict.fromkeys(list_dir(root=os.path.join(self.root, alphabet)))

            # loop through each alphabet
            for character in self.data[alphabet]:
                self.data[alphabet][character] = []

                # loop through all images in an alphabet character
                for img_name in list_files(root=os.path.join(self.root, alphabet, character), suffix=suffix):
                    if self.load_images:
                        # load images
                        img = _load_image(img_url=os.path.join(self.root, alphabet, character, img_name), expand_dim=self.expand_dim)
                    else:
                        img = img_name

                    self.data[alphabet][character].append(img)

    def generate_episode(self, episode_name: typing.Optional[typing.List[str]] = None) -> typing.List[np.ndarray]:
        """Generate an episode of data and labels

        Args:
            episode_name (List(str)): list of string where the first one is
                the name of the alphabet, and the rest are character names.
                If None, it will randomly pick characters from the given alphabet
            num_imgs (int): number of images per character

        Returns:
            x (List(numpy array)): list of numpy array representing data
        """
        x = []

        if episode_name is None:

            alphabet = random.sample(population=self.data.keys(), k=1)[0]

            max_n_way = min(len(self.data[alphabet]), self.max_num_cls)

            assert self.min_num_cls <= max_n_way

            n_way = random.randint(a=self.min_num_cls, b=max_n_way)
            n_way = min(n_way, self.max_num_cls)

            characters = random.sample(population=self.data[alphabet].keys(), k=n_way)
        else:
            alphabet = episode_name[0]
            characters = episode_name[1:]

        for character in characters:
            x_temp = random.sample(population=self.data[alphabet][character], k=self.k_shot)
            if self.load_images:
                x.append(x_temp)
            else:
                x_ = [_load_image(
                    img_url=os.path.join(self.root, alphabet, character, img_name),
                    expand_dim=self.expand_dim
                ) for img_name in x_temp]
                x.append(x_)

        return x

class ImageFolderGenerator(object):
    def __init__(
        self,
        root: str,
        train_subset: bool = True,
        suffix: str = '.png',
        min_num_cls: int = 5,
        max_num_cls: int = 20,
        k_shot: int = 16,
        expand_dim: bool = False,
        load_images: bool = False,
        xml_url: typing.Optional[str] = None
    ) -> None:
        """Initialize a dataloader instance for image folder structure

        Args:
            root (str): location containing ``train`` and ``test`` folders,
                where each folder contains a number of image folders
            train_subset (bool): If True, take data from ``train`` folder,
                else ``test`` folder
            suffix (str): the suffix of all images
            min_num_cls (int): minimum number of classes to pick to form an episode
            max_num_cls (int): maximum number of classes to pick to form an episode
            expand_dim (bool): useful for black and white images only
                (convert from 1 channel to 3 channels)
            load_images (bool): load images to RAM. Set True if dataset is small
            xml_url (str): location of the XML structure

        """
        self.root = os.path.join(root, 'train' if train_subset else 'test')
        self.suffix = suffix
        self.k_shot = k_shot
        self.expand_dim = expand_dim
        self.load_images = load_images
        self.xml_url = xml_url

        self.data = self.get_data()

        assert min_num_cls <= len(self.data)
        self.min_num_cls = min_num_cls
        self.max_num_cls = min(max_num_cls, len(self.data))

    def get_data(self):
        """Get class-image data stored in a dictionary
        """
        data_str = get_cls_img(root=self.root, suffix=self.suffix)

        if not self.load_images:
            return data_str

        cls_img_data = dict.fromkeys(data_str.keys())
        for cls_ in data_str:
            temp = [0] * len(data_str[cls_])
            for i, img_name in enumerate(data_str[cls_]):
                img = _load_image(
                    img_url=os.path.join(self.root, cls_, img_name),
                    expand_dim=self.expand_dim
                )
                temp[i] = img
            cls_img_data[cls_] = list(temp)

        return cls_img_data

    def generate_episode(self, episode_name: typing.Optional[typing.List[str]] = None) -> typing.List[np.ndarray]:
        """Generate an episode
        Args:
            episode_name (str): a list of classes to form the episode.
                If None, sample a random list of classes.

        Returns:
            x (list(numpy array)): list of images loaded in numpy array form
        """
        x = []

        if episode_name is not None:
            cls_list = episode_name
        elif self.xml_url is None:
            cls_list = random.sample(
                population=self.data.keys(),
                k=random.randint(a=self.min_num_cls, b=self.max_num_cls)
            )
        else:
            raise NotImplementedError('Not implemeted yet')

        for cls_ in cls_list:
            x_temp = random.sample(population=self.data[cls_], k=self.k_shot)
            if self.load_images:
                x.append(x_temp)
            else:
                x_ = [_load_image(
                    img_url=os.path.join(self.root, cls_, img_name),
                    expand_dim=self.expand_dim
                ) for img_name in x_temp]
                x.append(x_)

        return x

class ImageEmbeddingGenerator(object):
    def __init__(
        self,
        root: str,
        train_subset: bool = True,
        suffix: str = '.pkl',
        min_num_cls: int = 5,
        max_num_cls: int = 20,
        k_shot: int = 16,
        expand_dim: bool = False,
        load_images: bool = True
    ) -> None:
        self.root = root
        self.suffix = suffix
        self.k_shot = k_shot
        self.expand_dim = expand_dim
        self.load_images = load_images
        self.train_subset = train_subset

        self.data = self.get_data()

        assert min_num_cls <= len(self.data)
        self.min_num_cls = min_num_cls
        self.max_num_cls = min(max_num_cls, len(self.data))
    
    def get_data(self) -> dict:
        cls_img_data = {}

        if self.train_subset:
            f_pkl = open(file=os.path.join(self.root, 'train.pkl'), mode='rb')
            all_classes, all_data = pickle.load(f_pkl)
            f_pkl.close()

            f_pkl = open(file=os.path.join(self.root, 'val.pkl'), mode='rb')
            all_class_val, all_data_val = pickle.load(f_pkl)
            f_pkl.close()

            all_classes.update(all_class_val)
            all_data.update(all_data_val)
        else:
            f_pkl = open(file=os.path.join(self.root, 'test.pkl'), mode='rb')
            all_classes, all_data = pickle.load(f_pkl)
            f_pkl.close()

        for key in all_classes:
            cls_img_data[key] = []
            for img_name in all_classes[key]:
                cls_img_data[key].append(all_data[img_name])
        
        return cls_img_data
    
    def generate_episode(self, episode_name: typing.Optional[typing.List[str]] = None) -> typing.List[np.ndarray]:
        x = []

        if episode_name is not None:
            cls_list = episode_name
        else:
            cls_list = random.sample(
                population=self.data.keys(),
                k=random.randint(a=self.min_num_cls, b=self.max_num_cls)
            )

        for cls_ in cls_list:
            x_temp = random.sample(population=self.data[cls_], k=self.k_shot)
            x.append(x_temp)

        return x