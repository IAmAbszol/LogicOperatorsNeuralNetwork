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
  <li>Optional: Add more operators within Logic.csv, <b>not</b> operator can work.</li>
  <li>run python train.py</li>
  <li>run python LogicNN.py -a/-o/-x to test specific operator. Add more for newer operators.</li>
</ol>

<h3>Notes</h3>
<p>
The newly trained network differs from the intial commit as training isn't segmented, rather, the training 
is now grouped together and identified by column index relative to the operators subset within the Logic.csv file.
</p>
