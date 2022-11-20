from collections import deque
import random
from AlphaZeroAgent import AlphaZeroAgent
from MCTSAgent import MCTSAgent
from game import Board, Game
from tqdm import tqdm

def update_agent(agent, train_batch, epochs):
    state_batch = [data[0] for data in train_batch];
    mcts_probs_batch = [data[1] for data in train_batch]
    winner_batch = [data[2] for data in train_batch]

    for i in range(epochs):
        loss, entropy, kl_div, lr_multiplier = agent.train_step(state_batch, mcts_probs_batch, winner_batch)
    print("kl:{}, lr_multiplier:{}, loss:{}, entropy:{},".format(kl_div, lr_multiplier, loss, entropy))
    return

def evaluate_agent(simulator, agent, eval_agent, eval_games):
    win_count = {1: 0, 2: 0, -1: 0}
    for i in tqdm(range(eval_games), desc="Evaluating"):
        winner = simulator.start_play(agent, eval_agent, start_player=i % 2, is_shown=0)
        win_count[winner] += 1
    win_ratio = (win_count[1] + 0.5 * win_count[-1]) / eval_games
    print("Win Ratio: {}".format(win_ratio))
    return win_ratio

def train(board_width=6, board_height=6, n_in_row=4, n_games=1500, batch_size=512, train_epochs=5, buffer_size=10000, n_playouts=400, eval_freq=50, eval_games=10, eval_playouts=1000, max_eval_playouts=5000, eval_playout_inc=1000):
    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
    simulator = Game(board)
    agent = AlphaZeroAgent(board, n_playouts=n_playouts, is_selfplay=True, model_file="models/best_policy_664_1.pt")
    eval_agent = MCTSAgent(n_playout=eval_playouts)
    
    data_buffer = deque(maxlen=buffer_size)

    best_win_ratio = 0

    for i in tqdm(range(n_games), desc="Training"):
        winner, game_data = simulator.start_self_play(agent)
        game_data = list(game_data)[:]
        data_buffer.extend(game_data)

        # print("Game: {}, Length: {}".format(i, len(game_data)))
        if len(data_buffer) > batch_size:
            train_batch = random.sample(data_buffer, batch_size)
            update_agent(agent, train_batch, train_epochs)

        if i % eval_freq == eval_freq - 1:
            print("Evaluating at game {}".format(i))
            agent.is_selfplay = False
            win_ratio = evaluate_agent(simulator, agent, eval_agent, eval_games)
            agent.save_model('./current_policy.pt')
            if win_ratio > best_win_ratio:
                print("New Best Policy")
                best_win_ratio = win_ratio
                agent.save_model('./best_policy.pt')
                # Improve the MCTS evaluator (by increasing number of playouts) if our model can win 100% of the time
                if best_win_ratio == 1.0 and eval_playouts < max_eval_playouts:
                    print("MCTS {} defeated. Upgrading MCTS...".format(eval_playouts))
                    eval_playouts += eval_playout_inc
                    best_win_ratio = 0.0
            agent.is_selfplay = True

if __name__ == "__main__":
    train()