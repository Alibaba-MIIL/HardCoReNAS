from collections import Iterable  # < py38

#TODO: Test
def first_occurrence(mylist: list, value):
    """
    Search for the index of the last occurrence of a value in the a list
    :param mylist: The list to search in
    :param value: The value to search for
    :return: The index of the last occurrence of a value in the a list
    """
    return mylist.index(value)

#TODO: Test
def last_occurrence(mylist: list, value):
    """
    Search for the index of the last occurrence of a value in the a list
    :param mylist: The list to search in
    :param value: The value to search for
    :return: The index of the last occurrence of a value in the a list
    """
    return len(mylist) - 1 - list.index(value)


#TODO: Test
def count_in_range(mylist: list, lb, ub):
    """
    Count the number of elements in the list in a given range, i.e. lb < x <= ub
    :param mylist: The list whose elements are to be counted
    :param lb: The lower bound of the range
    :param ub: The upper bound of the range
    :return: The number of elements in the list in a given range
    """
    return len([x for x in mylist if lb < x <= ub])

# TODO: Test
def flatten_nested_list(items):
    """
    Flatten a nested list in to a flat version of the given list with a single hierarchy
    :param items: The nested list to be flatten
    :return: A flat list
    :param items:
    :return:
    """
    return [x for x in flatten(items)]


# TODO: Test
def flatten(items: iter):
    """
    Yield items from any nested iterable;
    See Beazley, D. and B. Jones. Recipe 4.14, Python Cookbook 3rd Ed., O'Reilly Media Inc. Sebastopol, CA: 2013.
    :param items: The nested iterable to be flatten
    :return: An iterator over the flatten items
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def append_list_to_list_or_item(items1, items2):
    """
    Append a list or a single item to a list or a single item
    :param items1: The list (or item) to append to
    :param items2: The list (or item) to be appended
    :return: The appended list
    """
    if type(items1) is not list:
        items1 = [items1]
    if type(items2) is not list:
        items2 = [items2]

    return items1 + items2

def deepcopy_nested_dict(nested_dict_to_deepcopy: dict):
    """
    Deepcopy of a nested dictionary of two levels, e.g. {k1:{...}, k2:{...}, ..., kN:{...}}
    :param nested_dict_to_deepcopy: The nested dictionary to return a deepcopy of
    :return: A deepcopy of a nested dictionary
    """
    # Copy the upper level
    deepcopied_nested_dict = nested_dict_to_deepcopy.copy()

    # Coppy the lower level
    for k, d in nested_dict_to_deepcopy.items():
        assert type(d) is dict
        deepcopied_nested_dict[k] = d.copy()

    return deepcopied_nested_dict

def set_list_of_lists(list_of_lists):
    """
    Removes duplicated nested lists of a given list of lists, i.e. the set() equivalent for lists of lists
    :param list_of_lists: A list of lists to take the set() of
    :return: A set() equivalent to the given list of lists
    """
    return [list(item) for item in set(tuple(nested_list ) for nested_list in list_of_lists)]
