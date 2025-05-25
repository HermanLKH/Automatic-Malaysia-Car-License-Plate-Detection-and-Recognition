# config.py
# Global configuration constants
IMG_HEIGHT = 32
IMG_WIDTH = 100
NUM_FIDUCIAL = 20       # number of TPS control points
INPUT_CHANNEL = 1      # grayscale input
OUTPUT_CHANNEL = 512   # feature extractor output channels
HIDDEN_SIZE = 256      # BiLSTM hidden size
# CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/'
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/!' # added the ! to identify the new line
# CTC uses an extra blank token
# NUM_CLASSES = len(CHARACTERS) + 1
NUM_CLASSES = len(CHARACTERS) + 2 # +1 for CTC blank or [GO] and +1 for [s] in attention → 39
MAX_LABEL_LENGTH = 20  # maximum label length # 25 original