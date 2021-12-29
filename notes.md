#data format description

we start from the preprocessed dataset whose suffix is '.jsonl'. 
Every line contains a dict including following keys: id, context, question, answer, answer's start id, 
and answer's end id. Namely, we call such a line an item.

##Dataset, examples and field
All the items will be loaded and packaged into a torch Dataset as examples. The context and questions will be further
divided under word level and character level by the format provided in fields. But the partitioning will not be immediately
performed at the time items are loaded into the Dataset as examples. You should call a function intentionally to perform 
the partitioning.

## Batch
We will divide items into batches. Firstly, we will sort these items, and make the items who have similar 
context length be close to each other. After that, we load data batch by batch. Since items who have similar context 
length tend to be divided into the same batch, we don't need to do much padding, which saves space cost.

## Vocabulary 
We will separately build library for characters and words that appear in all the context and questions. Now, 
in a character library, every character is represented by a unique number. Similarly, every word is represented by a unique 
number in the word library.

#Padding
Suppose we set batch size as 5. Then we load 5 items each time. Suppose the longest context among these 5 items has a 
length of 10, and the longest word in contexts among all the items has a length of 8.
Then, every word will be padded by a unique padding token until it has a length of 8. Next, every context will be padded
as it has 10 words. Therefore, contexts become a numerical tensor whose size is ``[5, 10, 8]`` under the character level 
and ``[5, 10]`` under the word level, if ``batch_first=True``.
We will do the same thing to questions, but notice that this action is separated from that of contexts.


For the field 'word', we have set ``inclued_lengh=True``. As a result, The attribute 'c_word' and 'q_word' both become a 
tuple. The first element of this tuple is the tensor we described before, while the second one is a one-dimensional tensor
that records the real length (*i.e.*, how many words) of each context or question. The length of this element is obviously 
equal to the batch size.

#Functional and Modules
# torch.nn.functional and torch.nn.Module
their several methods have similar names and almost same functions. However, methods in 'functional' will perform the calculation directly, 
while methods in 'Module' will create a layer. Therefore, if you want the calculation be performed both when training and testing, then use 
the former one, otherwise, if you want it to function only when training, then choose the latter one. 

## nn.Embedding
at least 2 parameters : the amount of words(or characters), the dimension you want to map them into.

the input data size is free, and the output size will be one more dimension than input, which is the word(or character)
vector's dimension.

padding idx can also be assigned

## nn.Conv2d
at least 3 parameters: input channel size, output channel size, and kernel size(kernel size can be a multi-dimension tuple)

the input datasize should be ``[batch size, input channel size, rows, columns]``, ``input channel size``is not 
neglectable even if it is ``1``.

## torch.cat
This function has a parameter 'dim', indicating along which dimension the concatenation should be performed. 
In particular, if ``torch.cat dim=-1``, concatenation will be performed along the last dimension of the tensor.
