# server.py
import flwr as fl
from blockchain import Blockchain


class FederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.blockchain = Blockchain(difficulty=2, save_to_file=True)

    def aggregate_fit(self, server_round, results, failures):
        # Use default FedAvg aggregation
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            print(f"Round {server_round}: Aggregated weights saved to blockchain.")

            # Mine a new block with the aggregated weights
            self.blockchain.mine_block(str(aggregated_weights))
        return aggregated_weights


def start_server():
    strategy = FederatedStrategy()
    fl.server.start_server(
        server_address="[::]:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )


if __name__ == "__main__":
    start_server()
