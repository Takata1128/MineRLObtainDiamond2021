def recursive_unzip(dictionary, item_idx):
    ret = {}
    for key, item in dictionary.items():
        if isinstance(item, dict):
            ret[key] = recursive_unzip(item, item_idx)
        else:
            ret[key] = item[item_idx]
    return ret


def unzip_states_or_actions(state_or_action_dict):
    ret = []
    num_items = len(state_or_action_dict.get("pov", state_or_action_dict.get("sneak")))
    for i in range(num_items):
        ret.append(recursive_unzip(state_or_action_dict, i))
    return ret
