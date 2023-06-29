def split_text(text, max_words=512):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    split_points = [i for i, word in enumerate(words) if word.endswith(('.', '!', '?'))]
    if not split_points:
        return [text[:max_words]]
    
    best_split = 0
    for i, split_point in enumerate(split_points):
        if split_point <= max_words:
            best_split = i
        else:
            break
    
    split_index = split_points[best_split]
    return [text[:split_index + 1]] + split_text(text[split_index + 1:], max_words)



