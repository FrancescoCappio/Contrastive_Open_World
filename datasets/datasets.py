import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import Sampler
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
import random

from utils.utils import set_random_seed


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
        random.shuffle(iterable)

def get_train_transform(resize_factor):

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(scale=(resize_factor,1.0),size=64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    return MultiDataTransform(train_transform)

def get_test_transform_crop():
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    return test_transform

def get_test_transform(square=False):
    resize_transform = transforms.Resize(64)
    if square:
        resize_transform = transforms.Resize((64,64))
    test_transform = transforms.Compose([
        resize_transform,
        transforms.ToTensor(),
    ])
    return test_transform

def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

class FileDataset(data.Dataset):
    def __init__(self, data_path, benchmark, data_file, transform=None,add_idx=False):
        super().__init__()

        self.root_dir = data_path
        self.benchmark = benchmark
        self.data_file = data_file
        self.transform = transform
        self.names, self.labels = _dataset_info(self.data_file)
        self.add_idx = add_idx

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        path, target = self.names[index],self.labels[index]
        orig_path = path

        path = os.path.expanduser(self.root_dir+f"{self.benchmark}/{path}")
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        img_size = img.size

        if self.transform is not None:
            img = self.transform(img)

        if self.add_idx:
            return img, target, index, orig_path
        return img, target

class ClassDataset(data.Dataset):
    def __init__(self, root, names, label, transform, add_index=False):
        super().__init__()

        self.root = root
        self.names = names
        self.label = label
        self.transform = transform
        self.add_index = add_index

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        path, target = self.names[index], self.label

        path = os.path.expanduser(self.root+f"/{path}")
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        img_size = img.size

        if self.transform is not None:
            img = self.transform(img)

        if self.add_index:
            return img, target, index, path
        return img, target, path

def get_pseudo_targets(P, known_mask, known_pseudo_labels, transform, n_classes):
    file_path = f'data/data_txt/{P.dataset}/{P.test_domain}.txt'

    names, _ = _dataset_info(file_path)
    np_names = np.array(names)

    selected_names = np_names[known_mask]
    selected_labels = known_pseudo_labels[known_mask]

    return get_class_datasets(P.data_path, P.dataset, data_file=None, transform=transform, names=selected_names.tolist(), labels = selected_labels.tolist(), n_classes=n_classes)


def get_class_datasets(data_path, benchmark, data_file, classes, transform=None, names=None, labels=None, train=False):
    # from a dataset file it builds a set of datasets, one for each class, keep only specificed classes 
    if names is None:
        names, labels = _dataset_info(data_file)

    selected_names, selected_labels = [], []
    np_labels = np.array(labels)
    np_names = np.array(names)
    all_classes = classes
    for lbl in all_classes:
        mask = np_labels == lbl
        selected_names.extend(np_names[mask].tolist())
        selected_labels.extend(np_labels[mask].tolist())

    root_dir = data_path

    names = np.array(selected_names)
    labels_set = set(selected_labels)
    labels = np.array(selected_labels)

    if len(labels_set) != len(classes):
        print("One dataset does not contain all the classes!")

    datasets = {}
    for lbl in labels_set:
        mask = labels == lbl
        class_names = names[mask]
        ds = ClassDataset(root_dir, class_names, lbl, transform, add_index=not train)
        datasets[lbl] = ds
    return datasets

def get_datasets_for_test(P):

    if "deit" in P.model:
        test_transform = get_test_transform(square=True)
    else:
        test_transform = get_test_transform()

    # target
    benchmark = P.dataset
    file_path = f'data/data_txt/{benchmark}/{P.test_domain}.txt'
    target_ds = FileDataset(P.data_path, benchmark, file_path, test_transform, add_idx=True)
    
    # source
    if benchmark == "OfficeHome":
        source_name = f"no_{P.test_domain}OpenSet"
        n_classes = 45
    elif benchmark == "Office31":
        source_name = f"no_{P.test_domain}OpenSet"
        n_classes = 20
    elif benchmark == "DomainNet":
        source_name = "OpenSet_source_train"
        n_classes = 100
    elif benchmark == "visda21":
        source_name = "imagenetOpenSet_known"
        n_classes = 1000
    else:
        raise NotImplementedError(f"Unknown benchmark {benchmark}")

    source_file_path = f'data/data_txt/{benchmark}/{source_name}.txt'

    source_ds = FileDataset(P.data_path, benchmark, source_file_path, test_transform, add_idx=True)

    return source_ds, target_ds, n_classes

def get_test_datasets(P, episode=0):

    test_transform = get_test_transform()

    if P.dataset == "OWR":
        classes_order_file = 'data/data_txt/OWR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]

        target_file = f'data/data_txt/OWR/{P.target}_test.txt'
        tot_known_classes = [11, 16, 21, 26][episode]
        selected_known_classes = selected_order[:tot_known_classes]

        tot_unknown_classes = 25
        selected_unknown_classes = selected_order[-(tot_unknown_classes):]

        known_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_known_classes, test_transform, train=False)
        unknown_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_unknown_classes, test_transform, train=False)
        # we should load the target dataset filtering only known + unknown classes

    elif P.dataset == "COSDA-HR":
        classes_order_file = 'data/data_txt/COSDA-HR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]

        target_file = f'data/data_txt/COSDA-HR/target.txt'
        tot_known_classes = [el for el in range(10,161,10)][episode]
        selected_known_classes = selected_order[:tot_known_classes]

        tot_unknown_classes = 1
        selected_unknown_classes = selected_order[-(tot_unknown_classes):]

        known_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_known_classes, test_transform, train=False)
        unknown_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_unknown_classes, test_transform, train=False)

    elif P.dataset == "CORe50":
        classes_order_file = 'data/data_txt/CORe50/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]

        target_file = f'data/data_txt/CORe50/target.txt'
        tot_known_classes = [10,15,20,25][episode]
        selected_known_classes = selected_order[:tot_known_classes]

        tot_unknown_classes = 25
        selected_unknown_classes = selected_order[-(tot_unknown_classes):]

        known_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_known_classes, test_transform, train=False)
        unknown_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_unknown_classes, test_transform, train=False)

        print(f"Known classes: {selected_known_classes}")
        print(f"Unknown classes: {selected_unknown_classes}")

    elif P.dataset == "OWR_val":
        classes_order_file = 'data/data_txt/OWR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]

        target_file = f'data/data_txt/OWR/{P.target}_test.txt'
        tot_known_classes = [9][episode]
        selected_known_classes = selected_order[:tot_known_classes]

        tot_unknown_classes = 2
        selected_unknown_classes = selected_order[-(tot_unknown_classes+1):]

        known_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_known_classes, test_transform, train=False)
        unknown_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_unknown_classes, test_transform, train=False)

    elif P.dataset == "OWR_val_inc":
        classes_order_file = 'data/data_txt/OWR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]

        target_file = f'data/data_txt/OWR/{P.target}_test.txt'
        tot_known_classes = [5, 9][episode]
        selected_known_classes = selected_order[:tot_known_classes]

        tot_unknown_classes = 2
        selected_unknown_classes = selected_order[-(tot_unknown_classes+1):]

        known_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_known_classes, test_transform, train=False)
        unknown_classes_datasets = get_class_datasets(P.data_path, P.dataset, target_file, selected_unknown_classes, test_transform, train=False)

    else:
        raise NotImplementedError(f"Dataset {P.dataset} is not known")

    return ConcatDataset([ds for cls, ds in known_classes_datasets.items()]), ConcatDataset([ds for cls, ds in unknown_classes_datasets.items()])

def get_dataset_2(P, episode=0, train=True, target_known_mask=None, target_known_pseudo_labels=None):

    if train:
        transform = get_train_transform(P.resize_factor)
    else:
        transform = get_test_transform(square=True)

    benchmark = P.dataset

    domain_datasets = {}
    if benchmark == "OWR":
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/OWR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [11, 16, 21, 26]
        classes_prec_episode = 0
        classes_till_now = tot_n_classes[episode]

        if episode > 0:
            classes_prec_episode = tot_n_classes[episode-1]
        n_classes_this_episode = classes_till_now - classes_prec_episode

        selected_classes = selected_order[classes_prec_episode:classes_till_now]

        P.classes_current_episode = selected_classes
        P.all_learned_classes = selected_order[:classes_till_now]

    elif benchmark == "COSDA-HR":
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/COSDA-HR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [el for el in range(10,161,10)]
        classes_prec_episode = 0
        classes_till_now = tot_n_classes[episode]

        if episode > 0:
            classes_prec_episode = tot_n_classes[episode-1]
        n_classes_this_episode = classes_till_now - classes_prec_episode

        selected_classes = selected_order[classes_prec_episode:classes_till_now]

        P.classes_current_episode = selected_classes
        P.all_learned_classes = selected_order[:classes_till_now]

    elif benchmark == "CORe50":
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/CORe50/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [10,15,20,25]
        classes_prec_episode = 0
        classes_till_now = tot_n_classes[episode]

        if episode > 0:
            classes_prec_episode = tot_n_classes[episode-1]
        n_classes_this_episode = classes_till_now - classes_prec_episode

        selected_classes = selected_order[classes_prec_episode:classes_till_now]

        P.classes_current_episode = selected_classes
        P.all_learned_classes = selected_order[:classes_till_now]


    elif benchmark == "OWR_val": # for validation protocol
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/OWR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [9]
        classes_prec_episode = 0
        classes_till_now = tot_n_classes[episode]

        if episode > 0:
            classes_prec_episode = tot_n_classes[episode-1]
        n_classes_this_episode = classes_till_now - classes_prec_episode

        selected_classes = selected_order[classes_prec_episode:classes_till_now]

        P.classes_current_episode = selected_classes
        P.all_learned_classes = selected_order[:classes_till_now]
        
    elif benchmark == "OWR_val_inc": # for validation protocol
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/OWR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [5, 9]
        classes_prec_episode = 0
        classes_till_now = tot_n_classes[episode]

        if episode > 0:
            classes_prec_episode = tot_n_classes[episode-1]
        n_classes_this_episode = classes_till_now - classes_prec_episode

        selected_classes = selected_order[classes_prec_episode:classes_till_now]

        P.classes_current_episode = selected_classes
        P.all_learned_classes = selected_order[:classes_till_now]

    else:
        raise NotImplementedError(f"Unknown benchmark {benchmark}")

    source_file = f'data/data_txt/{benchmark}/{source}_train.txt'

    # we should load only the classes from current order and episode

    datasets = get_class_datasets(P.data_path, benchmark, source_file, selected_classes, transform, train=train)

    # single source only for now
    domain_datasets[source] = datasets

    """
    if target_known_mask is not None: 
        domain_datasets["pseudo_target"] = get_pseudo_targets(P, target_known_mask, target_known_pseudo_labels, transform, n_classes=n_classes)
        sources.append("pseudo_target")
    """
    
    # now for each class we build a ConcatDataset
    class_datasets = []
    for idx in selected_classes:
        this_class = []
        for source in sources:
            if idx in domain_datasets[source]:
                this_class.append(domain_datasets[source][idx])
        class_datasets.append(this_class)

    class_datasets = [ConcatDataset(sets) for sets in class_datasets]
    return class_datasets

def get_replay_datasets(P, episode, selected_ids_dict):
    """
    This creates class datasets for old classes (episodes before current one), then it subsamples them with selected ids
    """
    if episode == 0:
        return []
    
    transform = get_test_transform(square=True)
    benchmark = P.dataset

    domain_datasets = {}
    if benchmark == "OWR":
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/OWR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [11, 16, 21, 26]

        n_old_classes = tot_n_classes[episode-1]
        old_classes = selected_order[:n_old_classes]

    elif benchmark == "COSDA-HR":
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/COSDA-HR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [el for el in range(10,161,10)]

        n_old_classes = tot_n_classes[episode-1]
        old_classes = selected_order[:n_old_classes]

    elif benchmark == "CORe50":
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/CORe50/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [10,15,20,25]

        n_old_classes = tot_n_classes[episode-1]
        old_classes = selected_order[:n_old_classes]

    elif benchmark == "OWR_val_inc":
        source = P.source
        sources = [source]

        classes_order_file = 'data/data_txt/OWR/fixed_order.npy'
        orders = np.load(classes_order_file)
        selected_order = orders[P.dataorder]
        
        tot_n_classes = [5,9]

        n_old_classes = tot_n_classes[episode-1]
        old_classes = selected_order[:n_old_classes]

    else:
        raise NotImplementedError(f"Unknown benchmark {benchmark}")

    source_file = f'data/data_txt/{benchmark}/{source}_train.txt'
    # we should load only the classes from current order and episode
    datasets = get_class_datasets(P.data_path, benchmark, source_file, old_classes, transform, train=False)

    # single source only for now
    domain_datasets[source] = datasets

    # now for each class we build a ConcatDataset and then subsample it
    class_datasets = []
    for idx in old_classes:
        this_class = []
        for source in sources:
            if idx in domain_datasets[source]:
                this_class.append(domain_datasets[source][idx])
        class_datasets.append(Subset(ConcatDataset(this_class), indices=selected_ids_dict[idx]))

    class_datasets = [ConcatDataset([sets]) for sets in class_datasets]

    # subsample classes
    return class_datasets


class DistributedMultiSourceRandomSampler(Sampler):
    r"""Samples elements randomly from a ConcatDataset, cycling between sources.
        Always with replacement, since batch should be balanced
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, num_samples=None, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank

        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.data_source, ConcatDataset):
            raise ValueError('data_source should be instance of ConcatDataset')

        self.cumulative_sizes = self.data_source.cumulative_sizes

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        indexes = []
        low = 0
        for i in range(len(self.cumulative_sizes)):
            high = self.cumulative_sizes[i]
            data_idx = torch.randint(low=low, high=high, size=(self.num_samples,), dtype=torch.int64).tolist()
            indexes.append(data_idx)
            low = high
        interleave_indexes = [x for t in zip(*indexes) for x in t]
        interleave_indexes = interleave_indexes[self.rank:self.cumulative_sizes[-1]:self.num_replicas] # distributed
        return iter(interleave_indexes)

    def __len__(self):
        return self.num_samples

class BalancedMultiSourceRandomSampler(Sampler):
    r"""Samples elements randomly from a ConcatDataset, cycling between sources.
        Designed to balance both classes and sources.
        Always with replacement, since batch should be balanced
    Arguments:
        data_source (Dataset): ConcatDataset of ConcatDatasets. First level: one concatdataset per class. Second level: one dataset for each source that has this class
        batch_p: number of samples of the same class that form a group. The group is the atomic batch size. E.g.: each gpu at each iteration receive at least a group
        rank: identifier of this GPU (for the separation of sampled elements between gpus for distributed training)
        world_size: number of GPUs taking part to distributed training
    """

    def __init__(self, data_source, batch_p, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.batch_p = batch_p

        if not isinstance(data_source, ConcatDataset):
            raise ValueError('data_source should be instance of ConcatDataset')

        for el in data_source.datasets:
            if not isinstance(el, ConcatDataset):
                raise ValueError("data_source should be a concatdataset of concat datasets")

        n_classes = len(data_source.datasets)
        sources_per_class = {}
        lengths_dict = {}
        ids_dict = {}

        low = 0
        for cls_id in range(n_classes):
            cls_ds = data_source.datasets[cls_id]
            # number of sources for this class
            n_sources = len(cls_ds.cumulative_sizes)
            sources_per_class[cls_id] = n_sources

            # size of each source for this class
            sizes = [cls_ds.cumulative_sizes[0]]
            sizes.extend([(cls_ds.cumulative_sizes[el] - cls_ds.cumulative_sizes[el-1]) for el in range(1,n_sources)])
            lengths_dict[cls_id] = sizes

            # elements of each source for this class -> taken in random order!
            low_this = 0

            src_dict = {}
            for src in range(n_sources):
                high = cls_ds.cumulative_sizes[src]
                ids_src = [el for el in range(low+low_this, high+low)]

                # random order
                random.shuffle(ids_src)
                src_dict[src] = ids_src

                low_this = high
            ids_dict[cls_id] = src_dict
            low += high

        list_strs, list_indices = BalancedMultiSourceRandomSampler.generate_list(n_classes, sources_per_class, batch_p, lengths_dict, ids_dict)

        # now we split indices among processes
        num_chunks = int(len(list_indices)/self.batch_p)

        while not num_chunks%self.world_size == 0:
            print("Removing some data as dataset size is not divisible per number of processes")
            for _ in range(batch_p):
                list_indices.pop()
            num_chunks = int(len(list_indices)/self.batch_p)

        indices_tensor = torch.tensor(list_indices)
        chunks = torch.chunk(indices_tensor, num_chunks)

        starts_from = self.rank
        my_chunks = []
        for idx, ch in enumerate(chunks):
            if (idx - starts_from)%self.world_size == 0:
                my_chunks.append(ch)
        my_indices = torch.cat(my_chunks).tolist()
        self.indices = my_indices

    @staticmethod
    def generate_list(n_classes, sources_per_class, batch_p, lengths_dict, ids_dict):
        """
        n_classes -> total number of classes
        sources_per_class -> dict with {class_id : number of sources containing this class}
        batch_p -> number of samples for each class in a block
        lengths_dict -> number of samples in each class for each source. Ex class 0 has K=sources_per_class[0] sources. lengths[0] = [len_class_0_source_1, ..., len_class_0_source_K] 
        ids_dict -> each sample has a unique identifier. This identifiers are those that should be inserted in the final list. This is a dict that contains for 
        each class and each souce a list of ids
        Should return a list which can be divided in blocks of size batch_p. Each block contains batch_p 
        elements of the same class. Subsequent blocks refer to different classes. 
        The sampling should always be done with replacement in order to maintain balancing. In particular
         - if for a certain class one source has less samples than the others those samples should be selected
         more often in order to rebalance the various sources;
         - if a certain class has in total a lower number of samples w.r.t. the others it should still appear in the 
         same number of blocks. 
         Therefore the correct approach is:
          - we compute the number of samples that we need for each class (max of each class number of sources*max_source_length)
          - for each class we randomly sample from the various sources (in an alternating fashion) until we reach the desired length
        Example of result with:
         - n_classes = 6
         - sources_per_class = {0:5,1:5,2:5,3:5,4:5,5:5} # -> each class has 5 sources
         - batch_p = 3
         - lengths_dict = {0:[8,8,8,8,8],1:[8,8,8,8,8],2:[8,8,8,8,8],3:[8,8,8,8,8],4:[8,8,8,8,8],5:[8,8,8,8,8]
         - ids_dict = {}
        OUTPUT: [
        D0C0E0, D1C0E0, D2C0E0,
        D0C1E0, D1C1E0, D2C1E0,
        D0C2E0, D1C2E0, D2C2E0,
        D0C3E0, D1C3E0, D2C3E0,
        D0C4E0, D1C4E0, D2C4E0,
        D0C5E0, D1C5E0, D2C5E0,
        D3C0E0, D4C0E0, D0C0E1,
        D3C1E0, D4C1E0, D0C1E1,
        D3C2E0, D4C2E0, D0C2E1,
        D3C3E0, D4C3E0, D0C3E1,
        D3C4E0, D4C4E0, D0C4E1,
        D3C5E0, D4C5E0, D0C5E1,
        D1C0E1, D2C0E1, D3C0E1,
        D1C1E1, D2C1E1, D3C1E1,
        D1C2E1, D2C2E1, D3C2E1,
        D1C3E1, D2C3E1, D3C3E1,
        D1C4E1, D2C4E1, D3C4E1,
        D1C5E1, D2C5E1, D3C5E1,
        D4C0E1, D0C0E2, D1C0E2,
        D4C1E1, D0C1E2, D1C1E2,
        D4C2E1, D0C2E2, D1C2E2,
        D4C3E1, D0C3E2, D1C3E2,
        D4C4E1, D0C4E2, D1C4E2,
        D4C5E1, D0C5E2, D1C5E2,
        ...
        ]
        First of all we compute the desired length for each class queue. 
        So for each class we compute num_sources*len_largest_source and we get the max of those values
        Then we create a queue for each class with the desired length and alternating samples from the various
        sources.
        We first create some intermediate parts that will help in finalizing the last list
        first for each source for each class we create a queue of elements:
        queue_C0_D0: [E0,E1,E2,E3,E4,E5,E6,E7]
        queue_C0_D1: [E0,E1,E2,E3,E4,E5,E6,E7]
        ...
        here we should take into account that sources should be balanced and therefore for those sources
        having a lower number of sample w.r.t. the others we will perform replacement
        Then for each class we create a queue that contains elements of that class alternating sources
        queue_C0 = [D0E0, D1E0, D2E0, D3E0, D4E0, D0E1, D1E1, D2E1, D3E1, D4E1, D0E2, D1E2, ...
        At this point we have a queue for each class. However it is possible that some queues are longer than others.
        Through resampling we should fix this so that we can keep the balancing between classes. 
        When resampling we should keep the alternating strategy for sources.
        """

        # compute desired length 
        cls_sizes = [max(lengths_dict[cls_id])*sources_per_class[cls_id] for cls_id in range(n_classes)]

        max_size = max(cls_sizes)

        desired_class_len = max_size

        # we duplicate each data structure
        # simply queues contains strings -> each string tell us how an element was chosen
        # while queues_ids contains the real ids
        queues = {}
        queues_ids = {}

        for cls_id in range(n_classes):

            n_sources = sources_per_class[cls_id]
            ids_this_class = ids_dict[cls_id]
            len_sources = lengths_dict[cls_id]
            queue_this_class = []
            queue_this_class_ids = []

            src_list = [idx for idx in range(n_sources)]
            random.shuffle(src_list)
            src_iter = iter(cycle(src_list))
            while len(queue_this_class) < desired_class_len:
                src = next(src_iter)
                ids_this_src = ids_this_class[src]
                len_this_src = len_sources[src]
                queue_this_class.append(f"D{src}E{random.randrange(len_this_src)}")
                queue_this_class_ids.append(ids_this_src[random.randrange(len_this_src)])

            queues[cls_id] = queue_this_class
            queues_ids[cls_id] = queue_this_class_ids

        out = []
        out_ids = []

        while True:
            found = False
            for cls_id in range(n_classes):
                q_this_class = queues[cls_id]
                q_this_class_ids = queues_ids[cls_id]
                if len(q_this_class) >= batch_p:
                    found = True
                    for el in range(batch_p):
                        out.append(f'C{cls_id}{q_this_class.pop(0)}')
                        out_ids.append(q_this_class_ids.pop(0))
            if not found:
                break
        return out, out_ids

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
