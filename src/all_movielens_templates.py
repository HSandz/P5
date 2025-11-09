'''
Pretraining Tasks for MovieLens -- 3 Main Prompt Families
1. Rating Prediction
2. Sequential Recommendation
3. Traditional Recommendation
'''

all_tasks = {}

# =====================================================
# Task Subgroup 1 -- Rating -- 10 Prompts
# =====================================================

task_subgroup_1 = {}

template = {}

'''
Input template:
Which star rating will user {{user_id}} give item {{item_id}}? (1 being lowest and 5 being highest)

Target template:
{{star_rating}}

Metrics:
Accuracy
'''

template['source'] = "Which star rating will user_{} give item_{} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-1"

task_subgroup_1["1-1"] = template


template = {}
'''
Input template:
How will user {{user_id}} rate this movie: {{item_title}}? (1 being lowest and 5 being highest)

Target template:
{{star_rating}}

Metrics:
Accuracy
'''
template['source'] = "How will user_{} rate this movie : {} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-2"

task_subgroup_1["1-2"] = template


template = {}
'''
Input template:
Will user {{user_id}} give item {{item_id}} a {{star_rating}}-star rating? (1 being lowest and 5 being highest)
 
Target template:
{{answer_choices[label]}} (yes/no)
 
Metrics:
Accuracy
'''
template['source'] = "Will user_{} give item_{} a {}-star rating ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'star_rating']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3"

task_subgroup_1["1-3"] = template


template = {}
'''
Input template:
Does user {{user_id}} like or dislike item {{item_id}}?

Target template:
{{answer_choices[label]}} (like/dislike) – like (4,5) / dislike (1,2,3)
 
Metrics:
Accuracy
'''
template['source'] = "Does user_{} like or dislike item_{} ?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike']
template['id'] = "1-4"

task_subgroup_1["1-4"] = template


template = {}
'''
Input template:
Does user {{user_id}} like item {{item_id}}?

Target template:
{{answer_choices[label]}} (yes/no) – yes (4,5) / no (1,2,3)

Metrics:
Accuracy
'''
template['source'] = "Does user_{} like item_{} ?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-5"

task_subgroup_1["1-5"] = template


template = {}
'''
Input template:
Predict the rating user {{user_id}} would give to the movie {{item_title}}. (1 being lowest and 5 being highest)

Target template:
{{star_rating}}

Metrics:
Accuracy
'''
template['source'] = "Predict the rating user_{} would give to the movie {} . ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-6"

task_subgroup_1["1-6"] = template


template = {}
'''
Input template:
Rate the movie {{item_title}} for user {{user_id}}. (1 being lowest and 5 being highest)

Target template:
{{star_rating}}

Metrics:
Accuracy
'''
template['source'] = "Rate the movie {} for user_{} . ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['item_title', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-7"

task_subgroup_1["1-7"] = template


template = {}
'''
Input template:
Would user {{user_id}} enjoy {{item_title}}?

Target template:
{{answer_choices[label]}} (yes/no) – yes (4,5) / no (1,2,3)

Metrics:
Accuracy
'''
template['source'] = "Would user_{} enjoy {} ?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-8"

task_subgroup_1["1-8"] = template

all_tasks['rating'] = task_subgroup_1


# =====================================================
# Task Subgroup 2 -- Sequential -- 15 Prompts
# =====================================================

task_subgroup_2 = {}

# --- Subgroup 2-1: Direct Prediction (5 prompts) ---

template = {}
'''
Input template:
Given the following movie watching history of user {{user_id}}: {{history}}, predict next possible movie to be watched by the user.

Target template:
{{item_title}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "Given the following movie watching history of user_{} : {} , predict next possible movie to be watched by the user ."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'history_item_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-1"

task_subgroup_2["2-1"] = template


template = {}
'''
Input template:
User {{user_id}} has watched these movies: {{history}}. What movie will they watch next?

Target template:
{{item_title}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "User_{} has watched these movies : {} . What movie will they watch next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'history_item_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-2"

task_subgroup_2["2-2"] = template


template = {}
'''
Input template:
Based on the history {{history}}, which movie should user {{user_id}} watch next?

Target template:
{{item_title}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "Based on the history {} , which movie should user_{} watch next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['history_item_titles', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-3"

task_subgroup_2["2-3"] = template


template = {}
'''
Input template:
Predict the next movie for user {{user_id}} given their viewing sequence: {{history}}.

Target template:
{{item_title}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "Predict the next movie for user_{} given their viewing sequence : {} ."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'history_item_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-4"

task_subgroup_2["2-4"] = template


template = {}
'''
Input template:
Recommend the next movie for user {{user_id}} based on {{history}}.

Target template:
{{item_title}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "Recommend the next movie for user_{} based on {} ."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'history_item_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-5"

task_subgroup_2["2-5"] = template


# --- Subgroup 2-2: Choose from Candidates (5 prompts) ---

template = {}
'''
Input template:
User {{user_id}} has watched {{history}}. From these candidates: {{candidates}}, which will they watch next?

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "User_{} has watched {} . From these candidates : {} , which will they watch next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history_item_titles', 'candidate_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-6"

task_subgroup_2["2-6"] = template


template = {}
'''
Input template:
Given user {{user_id}}'s history {{history}}, choose the next movie from: {{candidates}}.

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "Given user_{}'s history {} , choose the next movie from : {} ."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history_item_titles', 'candidate_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-7"

task_subgroup_2["2-7"] = template


template = {}
'''
Input template:
Select from {{candidates}} the next movie for user {{user_id}} who watched {{history}}.

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "Select from {} the next movie for user_{} who watched {} ."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['candidate_titles', 'user_id', 'history_item_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-8"

task_subgroup_2["2-8"] = template


template = {}
'''
Input template:
Based on {{history}}, which movie from {{candidates}} should user {{user_id}} watch next?

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "Based on {} , which movie from {} should user_{} watch next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['history_item_titles', 'candidate_titles', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-9"

task_subgroup_2["2-9"] = template


template = {}
'''
Input template:
User {{user_id}}'s viewing history: {{history}}. Pick the next movie from: {{candidates}}.

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "User_{}'s viewing history : {} . Pick the next movie from : {} ."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history_item_titles', 'candidate_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "2-10"

task_subgroup_2["2-10"] = template


# --- Subgroup 2-3: Yes/No Question (5 prompts) ---

template = {}
'''
Input template:
Given user {{user_id}}'s history {{history}}, will they watch {{item_title}} next?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "Given user_{}'s history {} , will they watch {} next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history_item_titles', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-11"

task_subgroup_2["2-11"] = template


template = {}
'''
Input template:
Is {{item_title}} the next movie for user {{user_id}} after watching {{history}}?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "Is {} the next movie for user_{} after watching {} ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['item_title', 'user_id', 'history_item_titles']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-12"

task_subgroup_2["2-12"] = template


template = {}
'''
Input template:
After watching {{history}}, will user {{user_id}} watch {{item_title}}?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "After watching {} , will user_{} watch {} ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['history_item_titles', 'user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-13"

task_subgroup_2["2-13"] = template


template = {}
'''
Input template:
User {{user_id}} watched {{history}}. Is {{item_title}} a good next recommendation?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "User_{} watched {} . Is {} a good next recommendation ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history_item_titles', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-14"

task_subgroup_2["2-14"] = template


template = {}
'''
Input template:
Based on {{history}}, should user {{user_id}} watch {{item_title}} next?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "Based on {} , should user_{} watch {} next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['history_item_titles', 'user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-15"

task_subgroup_2["2-15"] = template

all_tasks['sequential'] = task_subgroup_2


# =====================================================
# Task Subgroup 5 -- Traditional Recommendation -- 10 Prompts
# =====================================================

task_subgroup_5 = {}

# --- Subgroup 5-1: Yes/No (5 prompts) ---

template = {}
'''
Input template:
Should we recommend movie {{item_title}} to user {{user_id}}?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "Should we recommend movie {} to user_{} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['item_title', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-1"

task_subgroup_5["5-1"] = template


template = {}
'''
Input template:
Is movie {{item_title}} a good recommendation for user {{user_id}}?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "Is movie {} a good recommendation for user_{} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['item_title', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-2"

task_subgroup_5["5-2"] = template


template = {}
'''
Input template:
Would user {{user_id}} be interested in {{item_title}}?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "Would user_{} be interested in {} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-3"

task_subgroup_5["5-3"] = template


template = {}
'''
Input template:
Recommend {{item_title}} to user {{user_id}}?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "Recommend {} to user_{} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['item_title', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-4"

task_subgroup_5["5-4"] = template


template = {}
'''
Input template:
Does user {{user_id}} match with movie {{item_title}}?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "Does user_{} match with movie {} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-5"

task_subgroup_5["5-5"] = template


# --- Subgroup 5-2: Choose from 100 candidates (5 prompts) ---

template = {}
'''
Input template:
From the candidate list {{candidates}}, choose the best movie for user {{user_id}}.

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "From the candidate list {} , choose the best movie for user_{} ."
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['candidate_titles', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "5-6"

task_subgroup_5["5-6"] = template


template = {}
'''
Input template:
Which movie from {{candidates}} would user {{user_id}} prefer?

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "Which movie from {} would user_{} prefer ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['candidate_titles', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "5-7"

task_subgroup_5["5-7"] = template


template = {}
'''
Input template:
Select a movie for user {{user_id}} from: {{candidates}}.

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "Select a movie for user_{} from : {} ."
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'candidate_titles']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "5-8"

task_subgroup_5["5-8"] = template


template = {}
'''
Input template:
Pick the most suitable movie from {{candidates}} for user {{user_id}}.

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "Pick the most suitable movie from {} for user_{} ."
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['candidate_titles', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "5-9"

task_subgroup_5["5-9"] = template


template = {}
'''
Input template:
Given candidates {{candidates}}, what is the best recommendation for user {{user_id}}?

Target template:
{{item_title}}

Metrics:
Accuracy
'''
template['source'] = "Given candidates {} , what is the best recommendation for user_{} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['candidate_titles', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_title']
template['id'] = "5-10"

task_subgroup_5["5-10"] = template

all_tasks['traditional'] = task_subgroup_5
