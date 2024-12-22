# blockchain.py
import hashlib
import time


class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = (
            f"{self.index}{self.timestamp}{self.data}{self.previous_hash}{self.nonce}"
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        while not self.hash.startswith("0" * difficulty):
            self.nonce += 1
            self.hash = self.calculate_hash()


class Blockchain:
    def __init__(self, difficulty=2, save_to_file=False):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty
        self.save_to_file = save_to_file

    def create_genesis_block(self):
        return Block(0, time.time(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        if self.save_to_file:
            self.save_block_to_file(new_block)

    def mine_block(self, data):
        new_block = Block(
            len(self.chain), time.time(), data, self.get_latest_block().hash
        )
        self.add_block(new_block)

    def save_block_to_file(self, block):
        with open(f"block_{block.index}.txt", "w") as f:
            f.write(f"Index: {block.index}\n")
            f.write(f"Timestamp: {block.timestamp}\n")
            f.write(f"Data: {block.data}\n")
            f.write(f"Previous Hash: {block.previous_hash}\n")
            f.write(f"Hash: {block.hash}\n")
