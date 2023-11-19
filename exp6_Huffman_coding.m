%Huffman Encoding

my_str = 'jason';

%calculation of probabilities
prob_dist = double(my_str);

%Display character vs. probability
disp('Character Probability:');
for i = 1:length(prob_dist)
	display(strcat(my_str(i),' -->  ',num2str(prob_dist(i))));
end
total = sum(prob_dist)

%initialize encoding array
for i = 1:length(my_str)
	sorted_str{i} = my_str(i);
end

% Save initial set of symbols and probabilities for later use
init_str = sorted_str;
init_prob = prob_dist;

%Huffman coding process
sorted_prob = prob_dist;
rear = 1;
while (length(sorted_prob) > 1)
	% Sort probs
	[sorted_prob,indeces] = sort(sorted_prob,'ascend');

	% Sort string based on indeces
	sorted_str = sorted_str(indeces);

	% Create new symbol
	new_node = strcat(sorted_str(2),sorted_str(1));
	new_prob = sum(sorted_prob(1:2));

	% Dequeue used symbols from "old" queue
	sorted_str =  sorted_str(3:length(sorted_str));
	sorted_prob = sorted_prob(3:length(sorted_prob));

	% Add new symbol back to "old" queue
	sorted_str = [sorted_str, new_node];
	sorted_prob = [sorted_prob, new_prob];

	% Add new symbol to "new" queue
	newq_str(rear) = new_node;
	newq_prob(rear) = new_prob;
	rear = rear + 1;
end

%Form Huffman tree

tree = [newq_str,init_str];
tree_prob = [newq_prob, init_prob];

% Sort all tree elements
[sorted_tree_prob,indeces] = sort(tree_prob,'descend');
sorted_tree = tree(indeces);

%Calculate tree parameters
parent(1) = 0;
num_children = 2;
for i = 2:length(sorted_tree)
	% Extract my symbol
	me = sorted_tree{i};

	% Find my parent's symbol (search until shortest match is found)
	count = 1;
	parent_maybe = sorted_tree{i-count};
	diff = strfind(parent_maybe,me);
	while (isempty(diff))
		count = count + 1;
		parent_maybe = sorted_tree{i-count};
		diff = strfind(parent_maybe,me);
	end
	parent(i) = i - count;
end

%plot Huffman tree
treeplot(parent);
title(strcat('Huffman Coding Tree - "',my_str,'"'));

%label tree nodes
[xs,ys,h,s] = treelayout(parent);
text(xs,ys,sorted_tree);

%label tree edges
for i = 2:length(sorted_tree)
	% Get my coordinate
	my_x = xs(i);
	my_y = ys(i);

	% Get parent coordinate
	parent_x = xs(parent(i));
	parent_y = ys(parent(i));

	% Calculate weight coordinate (midpoint)
	mid_x = (my_x + parent_x)/2;
	mid_y = (my_y + parent_y)/2;

	% Calculate weight (positive slope = 1, negative = 0)
	slope  = (parent_y - my_y)/(parent_x - my_x);
	if (slope > 0)
		weight(i) = 1;
	else
		weight(i) = 0;
	end
	text(mid_x,mid_y,num2str(weight(i)));
end

%print codebook
for i = 1:length(sorted_tree)
	% Initialize code
	code{i} = '';

	% Loop until root is found
	index = i;
	p = parent(index);
	while(p ~= 0)
		% Turn weight into code symbol
		w = num2str(weight(index));

		% Concatenate code symbol
		code{i} = strcat(w,code{i});

		% Continue towards root
		index = parent(index);
		p = parent(index);
	end
end
codeBook = [sorted_tree', code']

