"""
MovieLens-100k Data Preprocessing for P5
Converts MovieLens-100k dataset to P5 format
"""

from collections import defaultdict
import os
import torch
import random
import numpy as np
import json
import pickle
import gzip
from tqdm import tqdm

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Set seeds
seed = 2022
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print("="*50)
print("MovieLens-100k Preprocessing for P5")
print("="*50)

# Configuration
short_data_name = 'ml100k'
rating_score = 0.0  # Filter threshold
user_core = 5
item_core = 5

# Create output directory (data/ml100k/)
output_dir = os.path.join('data', short_data_name)
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}/")

# =====================================================
# Step 1: Load MovieLens-100k Data
# =====================================================

def load_movielens_ratings(rating_score=0.0):
    """
    Load MovieLens-100k ratings data
    Format: user_id \t item_id \t rating \t timestamp
    """
    print("\nLoading ratings from raw_data/ml-100k/u.data...")
    datas = []
    data_file = './raw_data/ml-100k/u.data'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}\nPlease run download_movielens script first!")
    
    with open(data_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user = parts[0]
            item = parts[1]
            rating = float(parts[2])
            timestamp = int(parts[3])
            
            if rating <= rating_score:  # Filter low ratings
                continue
            
            datas.append((user, item, timestamp))
    
    print(f"Loaded {len(datas)} interactions")
    return datas

def load_movielens_meta():
    """
    Load MovieLens-100k item metadata
    Format: movie_id | movie_title | release_date | video_release_date | IMDb_URL | genres
    """
    print("Loading movie metadata from raw_data/ml-100k/u.item...")
    meta_file = './raw_data/ml-100k/u.item'
    meta_data = {}
    
    genre_names = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    with open(meta_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = parts[0]
            title = parts[1]
            
            # Get genres (last 19 columns are binary genre indicators)
            genres = []
            for i, is_genre in enumerate(parts[5:]):
                if is_genre == '1':
                    genres.append(genre_names[i])
            
            meta_data[item_id] = {
                'title': title,
                'categories': [genres] if genres else [['unknown']]
            }
    
    print(f"Loaded metadata for {len(meta_data)} movies")
    return meta_data

# =====================================================
# Step 2: Core Processing Functions
# =====================================================

def get_interaction(datas):
    """Convert raw data to user sequences"""
    print("\nConverting to user sequences...")
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = [(item, time)]
    
    # Sort by timestamp
    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = [t[0] for t in item_time]
        user_seq[user] = items
    
    print(f"Created sequences for {len(user_seq)} users")
    return user_seq

def check_Kcore(user_items, user_core, item_core):
    """Check if data satisfies K-core property"""
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1
    
    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True

def filter_Kcore(user_items, user_core, item_core):
    """Iteratively filter data to satisfy K-core"""
    print(f"\nApplying {user_core}-core filtering for users and {item_core}-core for items...")
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    iteration = 0
    while not isKcore:
        iteration += 1
        # Use list() to avoid modifying dict during iteration (matches original Amazon code behavior)
        for user in list(user_items.keys()):
            if user_count[user] < user_core:
                user_items.pop(user, None)
            else:
                # Filter items - need to create new list to avoid modification during iteration
                user_items[user] = [item for item in user_items[user] if item_count[item] >= item_core]
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
        print(f"  Iteration {iteration}: {len(user_items)} users, {len(item_count)} items")
    
    print(f"K-core filtering complete!")
    return user_items

def id_map(user_items):
    """Create ID mappings for users and items"""
    print("\nCreating ID mappings...")
    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    user_id = 1
    item_id = 1
    final_data = {}
    
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        
        iids = []
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        
        uid = user2id[user]
        final_data[uid] = iids
    
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    
    print(f"Created mappings: {user_id-1} users, {item_id-1} items")
    return final_data, user_id-1, item_id-1, data_maps

# =====================================================
# Main Processing Pipeline
# =====================================================

# Load data
datas = load_movielens_ratings(rating_score=rating_score)
print(f"Raw data loaded! Filtered interactions with rating > {rating_score}")

# Get user-item interactions
user_items = get_interaction(datas)

# Apply K-core filtering
user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)

# Create ID mappings
user_items, user_num, item_num, data_maps = id_map(user_items)
user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)

# Statistics
print("\n" + "="*50)
print("Dataset Statistics")
print("="*50)
user_count_list = list(user_count.values())
user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
item_count_list = list(item_count.values())
item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
interact_num = np.sum(user_count_list)
sparsity = (1 - interact_num / (user_num * item_num)) * 100

print(f"Total Users: {user_num}")
print(f"  Avg interactions per user: {user_avg:.2f}")
print(f"  Min: {user_min}, Max: {user_max}")
print(f"\nTotal Items: {item_num}")
print(f"  Avg interactions per item: {item_avg:.2f}")
print(f"  Min: {item_min}, Max: {item_max}")
print(f"\nTotal Interactions: {interact_num}")
print(f"Sparsity: {sparsity:.2f}%")

# =====================================================
# Step 3: Save Sequential Data
# =====================================================

print("\n" + "="*50)
print("Saving Sequential Data")
print("="*50)

data_file = os.path.join(output_dir, 'sequential_data.txt')
with open(data_file, 'w') as out:
    for user, items in user_items.items():
        out.write(user + ' ' + ' '.join(items) + '\n')
print(f"✓ Saved: {data_file}")

datamaps_file = os.path.join(output_dir, 'datamaps.json')
with open(datamaps_file, 'w') as out:
    json.dump(data_maps, out)
print(f"✓ Saved: {datamaps_file}")

# =====================================================
# Step 4: Process and Save Metadata
# =====================================================

print("\n" + "="*50)
print("Processing Metadata")
print("="*50)

raw_meta = load_movielens_meta()

# Filter metadata to only include items in our dataset
meta_data = {}
for raw_item_id, info in raw_meta.items():
    if raw_item_id in data_maps['item2id']:
        item_id = data_maps['item2id'][raw_item_id]
        meta_data[item_id] = info

print(f"Filtered metadata to {len(meta_data)} items in dataset")

# Save as gzipped JSON (following P5 format)
meta_file = os.path.join(output_dir, 'meta.json.gz')
with gzip.open(meta_file, 'wt', encoding='utf-8') as f:
    json.dump(meta_data, f)
print(f"✓ Saved: {meta_file}")

# Create user_id2name mapping (user IDs map to themselves for MovieLens)
user_id2name = {user_id: user_id for user_id in data_maps['id2user'].values()}
user_id2name_file = os.path.join(output_dir, 'user_id2name.pkl')
save_pickle(user_id2name, user_id2name_file)
print(f"✓ Saved: {user_id2name_file}")

# =====================================================
# Step 5: Generate Negative Samples
# =====================================================

print("\n" + "="*50)
print("Generating Negative Samples for Testing")
print("="*50)

def sample_test_data(user_items, test_num=99, sample_type='random'):
    """
    Sample negative items for each user for testing
    sample_type: 'random' or 'pop'
    """
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            item_count[int(item)] += 1
    
    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum(count)
    probability = [value / sum_value for value in count]
    
    user_neg_items = {}
    
    for user, user_seq in tqdm(user_items.items(), desc="Sampling negatives"):
        user_seq_int = [int(i) for i in user_seq]
        test_samples = []
        while len(test_samples) < test_num:
            if sample_type == 'random':
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else:
                sample_ids = np.random.choice(all_item, test_num, replace=False, p=probability)
            sample_ids = [str(item) for item in sample_ids if item not in user_seq_int and str(item) not in test_samples]
            test_samples.extend(sample_ids)
        test_samples = test_samples[:test_num]
        user_neg_items[user] = test_samples
    
    return user_neg_items

user_neg_items = sample_test_data(user_items)

test_file = os.path.join(output_dir, 'negative_samples.txt')
with open(test_file, 'w') as out:
    for user, samples in user_neg_items.items():
        out.write(user + ' ' + ' '.join(samples) + '\n')
print(f"✓ Saved: {test_file} ({len(user_neg_items)} users)")

# =====================================================
# Step 6: Create Rating Splits
# =====================================================

print("\n" + "="*50)
print("Creating Rating Splits (Train/Val/Test)")
print("="*50)

def load_all_ratings():
    """Load all ratings (not filtered by score)"""
    rating_data = []
    data_file = './raw_data/ml-100k/u.data'
    
    with open(data_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user = parts[0]
            item = parts[1]
            rating = float(parts[2])
            timestamp = int(parts[3])
            
            # Only include users and items that are in our filtered dataset
            if user in data_maps['user2id'] and item in data_maps['item2id']:
                user_id = data_maps['user2id'][user]
                item_id = data_maps['item2id'][item]
                
                # Use Amazon-compatible field names for P5 compatibility
                rating_data.append({
                    'reviewerID': user,      # Keep original ID for lookup
                    'asin': item,            # Keep original ID for lookup  
                    'user_id': user_id,      # Mapped ID
                    'item_id': item_id,      # Mapped ID
                    'overall': rating,       # Rating field (Amazon uses 'overall')
                    'timestamp': timestamp
                })
    
    return rating_data

rating_data = load_all_ratings()
print(f"Loaded {len(rating_data)} rating records")

# Create train/val/test splits (80/10/10)
population = len(rating_data)
indices = list(range(population))
random.shuffle(indices)

# Ensure each user and item appears at least once in training
user_mention_dict = defaultdict(list)
item_mention_dict = defaultdict(list)

for i in indices:
    user = rating_data[i]['user_id']
    item = rating_data[i]['item_id']
    user_mention_dict[user].append(i)
    item_mention_dict[item].append(i)

# Add at least one sample per user and item to training
train_indices = set()
for user, idx_list in user_mention_dict.items():
    train_indices.add(random.choice(idx_list))
for item, idx_list in item_mention_dict.items():
    train_indices.add(random.choice(idx_list))

print(f"Initial train indices (coverage): {len(train_indices)}")

# Fill remaining to reach 80%
remaining_indices = list(set(indices) - train_indices)
random.shuffle(remaining_indices)

train_target = int(population * 0.8)
need_more = train_target - len(train_indices)
train_indices.update(remaining_indices[:need_more])
train_indices = list(train_indices)

# Split remaining into val/test
val_test_indices = list(set(indices) - set(train_indices))
random.shuffle(val_test_indices)

val_size = len(val_test_indices) // 2
val_indices = val_test_indices[:val_size]
test_indices = val_test_indices[val_size:]

print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

# Create split datasets
train_rating_data = [rating_data[i] for i in train_indices]
val_rating_data = [rating_data[i] for i in val_indices]
test_rating_data = [rating_data[i] for i in test_indices]

rating_splits = {
    'train': train_rating_data,
    'val': val_rating_data,
    'test': test_rating_data,
    'train_indices': train_indices,
    'val_indices': val_indices,
    'test_indices': test_indices
}

rating_splits_file = os.path.join(output_dir, 'rating_splits.pkl')
save_pickle(rating_splits, rating_splits_file)
print(f"✓ Saved: {rating_splits_file}")