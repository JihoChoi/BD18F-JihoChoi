# ================================================================================
#
#   BDDL 2018 HW 02 Script
#
# ================================================================================
#
#   BDDL 2018 HW 02 Distributed Deep Learning Training
#                           with Vanilla TensorFlow, Horovod, Parallax
#
#   Author: Jiho Choi (jihochoi@snu.ac.kr)
#
#   GAN model by Aymeric Damien
#       - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/gan.py
#
# ================================================================================

# --------------------------
# Eable localhost on AWS EC2
# --------------------------

# ssh-keygen -t rsa
#      Press enter for each line
# cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
# chmod og-wx ~/.ssh/authorized_keys

# --------------------------
# Start virtual enviornment
source ~/parallax_venv/bin/activate # deactivate
pip show tensorflow horovod parallax


# --------------------------------------
# GAN Original
# --------------------------------------



# --------------------------------------
# GAN Distributed TensorFlow
# --------------------------------------



python ~/parallax/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=tf_ckpt/model.ckpt-0 --tensor_name=conv1/kernel




# --------------------------------------
# GAN Horovod
# --------------------------------------


# --------------------------------------
# GAN Parallax
# --------------------------------------


python run_parallax.py --max_steps=10













