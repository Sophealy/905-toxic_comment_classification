import torch.nn as nn
from torch.nn import functional as F

import torch


# Code strongly inspired from https://github.com/prakashpandey9/Text-Classification-Pytorch/tree/master/models


class BiLSTM(nn.Module):
	def __init__(self, output_size, hidden_size, embedding_length):
		super(BiLSTM, self).__init__()

		"""
		Parameters:

		output_size :
		hidden_sie : Size of the hidden_state of the LSTM
		embedding_length : Embeddding dimension

		"""
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.embedding_length = embedding_length


		self.lstm = nn.LSTM(embedding_length, hidden_size,dropout = 1,num_layers = 2, bidirectional =True)
		self.label = nn.Linear(hidden_size, output_size)

	def forward(self, input_sentence):

		"""
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)

		Returns
		-------
		Output of the linear layer containing logits  which receives its input as the final_hidden_state of the LSTM

		"""

		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''

		output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence)
		final_output = self.label(final_hidden_state[-1])

		return final_output
class LSTMAttention(torch.nn.Module):
	def __init__(self, output_size, hidden_size, embedding_length,):
		super(LSTMAttention, self).__init__()

		"""
		Arguments
		---------
		output_size :
		hidden_sie : Size of the hidden_state of the LSTM
		embedding_length : Embeddding dimension

		--------
		"""

		self.output_size = output_size
		self.hidden_size = hidden_size
		self.embedding_length = embedding_length

		self.lstm = nn.LSTM(embedding_length, hidden_size)
		self.label = nn.Linear(hidden_size, output_size)
		#self.attn_fc_layer = nn.Linear()

	def attention_net(self, lstm_output, final_state):

		"""
		Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
		between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

		Arguments
		---------

		lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
		final_state : Final time-step hidden state (h_n) of the LSTM

		---------

		Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
				  new hidden state.

		Tensor Size :
					hidden.size() = (batch_size, hidden_size)
					attn_weights.size() = (batch_size, num_seq)
					soft_attn_weights.size() = (batch_size, num_seq)
					new_hidden_state.size() = (batch_size, hidden_size)

		"""

		hidden = final_state.squeeze(0)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

		return new_hidden_state

	def forward(self, input_sentences):

		"""
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

		Returns
		-------
		Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
		final_output.shape = (batch_size, output_size)

		"""


		output, (final_hidden_state, final_cell_state) = self.lstm(input_sentences) # final_hidden_state.size() = (1, batch_size, hidden_size)
		output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)

		attn_output = self.attention_net(output, final_hidden_state)
		logits = self.label(attn_output)

		return logits



class CNN(nn.Module):
	def __init__(self, output_size,kernel_heights,embedding_length, in_channels =1, out_channels=1 , keep_probab = 0.2):
		super(CNN, self).__init__()

		"""
		Arguments
		---------
		output_size : 6
		in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
		out_channels : Number of output channels after convolution operation performed on the input matrix
		kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
		keep_probab : Probability of retaining an activation node during dropout operation
		embedding_length : Embedding dimension
		--------

		"""
		self.output_size = output_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_heights = kernel_heights
		self.embedding_length = embedding_length


		self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length) )
		self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length))
		self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length))
		self.dropout = nn.Dropout(keep_probab)
		self.label = nn.Linear(len(kernel_heights)*out_channels, output_size)

	def conv_block(self, input, conv_layer):
		conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
		activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
		max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)

		return max_out

	def forward(self, input_sentences):

		"""
		The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix
		whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
		We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
		and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
		to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

		Parameters
		----------
		input_sentences: input_sentences of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

		Returns
		-------
		Output of the linear layer containing logits for pos & neg class.
		logits.size() = (batch_size, output_size)

		"""

		input = input_sentences.permute(1,0,2)


		input = input.unsqueeze(1)

		# input.size() = (batch_size, 1, num_seq, embedding_length)
		max_out1 =F.relu(self.conv_block(input, self.conv1)) 
		max_out2 =F.relu( self.conv_block(input, self.conv2))
		max_out3 =F.relu( self.conv_block(input, self.conv3))


		all_out = torch.cat((max_out1, max_out2, max_out3), 1)


		fc_in = self.dropout(all_out)
		# fc_in.size()) = (batch_size, num_kernels*out_channels)
		logits = self.label(fc_in)

		return logits
