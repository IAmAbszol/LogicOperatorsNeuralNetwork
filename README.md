# LogicOperatorsNeuralNetwork
A Neural Network that learns logic operators based on training sessions provided.

<h2>Dependencies</h2>
<ul>
  <li>Python3</li>
  <li>Tensorflow</li>
  <li>Numpy==1.14.5</li>
  <li>Pandas</li>
</ul>

<h2>How to run</h2>
<ol>
  <li>Unzip LogicNN.zip</li>
  <li>Optional: Create <b>conda</b> environment.</li>
  <li>python train.py -a/-o/-x to train specific operator</li>
  <li>python LogicNN.py -a/-o/-x to test specific operator</li>
</ol>

<h3>Notes</h3>
<ul>
  <li>Previous train.py was ran to train <b>all</b> operators but it seems tensorflows library is separate from the program, possibly causing it to fail the training session. The first operator will be trained then the following are incorrect, swap the commenting to understand more on this issue. (Swap for and if True)</li>
</ul>
