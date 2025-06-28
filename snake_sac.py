import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
GRID_SIZE = 40
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
FPS = 120

BACKGROUND = (15, 20, 25)
GRID_COLOR = (30, 35, 40)
SNAKE_HEAD = (50, 180, 255)
SNAKE_BODY = (40, 120, 200)
FOOD_COLOR = (255, 80, 80)
TEXT_COLOR = (220, 220, 220)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 160, 210)
RED = (255, 60, 60)
GREEN = (50, 200, 100)
PURPLE = (180, 100, 220)
WALL_COLOR = (100, 100, 120)
INFO_BG = (25, 30, 35, 200)


UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(f"AI 贪吃蛇 - SAC 算法  {device}")
clock = pygame.time.Clock()

font_small = pygame.font.SysFont('OPPO Sans 4.0', 16)
font_medium = pygame.font.SysFont('OPPO Sans 4.0', 24)
font_large = pygame.font.SysFont('OPPO Sans 4.0', 32)


class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER
        self.is_hovered = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2, border_radius=8)

        text_surf = font_medium.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)

    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
       
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
  
        self.direction = RIGHT
        self.next_direction = RIGHT
     
        self.food = self.generate_food()
   
        self.score = 0
        self.steps = 0
        self.game_over = False
    
        self.snake_length = 1
     
        self.last_reward = 0
       
        self.move_history = []

    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def step(self, action=None):
        if self.game_over:
            return None, None, True

        self.steps += 1

    
        if action is not None:
            directions = [UP, RIGHT, DOWN, LEFT]
            new_direction = directions[action]

          
            if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
                self.next_direction = new_direction

        self.direction = self.next_direction
        self.move_history.append(self.direction)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT:
            self.game_over = True
            reward = -10  
            self.last_reward = reward
            return self.get_state(), reward, self.game_over

   
        if new_head in self.snake:
            self.game_over = True
            reward = -10  
            self.last_reward = reward
            return self.get_state(), reward, self.game_over

        self.snake.insert(0, new_head)

    
        if new_head == self.food:
            self.score += 1
            self.snake_length += 1
            self.food = self.generate_food()
            reward = 10  
            self.steps = 0 
        else:
           
            self.snake.pop()
            
            reward = 0.1
            if self.steps > 100:
                reward = -0.5

        if self.steps > 200:
            self.game_over = True
            reward = -5

        self.last_reward = reward
        return self.get_state(), reward, self.game_over

    def get_state(self):
        head_x, head_y = self.snake[0]

        food_dx = self.food[0] - head_x
        food_dy = self.food[1] - head_y

        danger_up = 0
        danger_right = 0
        danger_down = 0
        danger_left = 0

        wall_up = head_y / GRID_HEIGHT
        wall_right = (GRID_WIDTH - head_x - 1) / GRID_WIDTH
        wall_down = (GRID_HEIGHT - head_y - 1) / GRID_HEIGHT
        wall_left = head_x / GRID_WIDTH

        directions = [UP, RIGHT, DOWN, LEFT]
        for i, (dx, dy) in enumerate(directions):
            next_pos = (head_x + dx, head_y + dy)

            if next_pos[0] < 0 or next_pos[0] >= GRID_WIDTH or next_pos[1] < 0 or next_pos[1] >= GRID_HEIGHT:
                if i == 0:
                    danger_up = 1
                elif i == 1:
                    danger_right = 1
                elif i == 2:
                    danger_down = 1
                elif i == 3:
                    danger_left = 1
       
            elif next_pos in self.snake[1:]:
                if i == 0:
                    danger_up = 1
                elif i == 1:
                    danger_right = 1
                elif i == 2:
                    danger_down = 1
                elif i == 3:
                    danger_left = 1


        dir_up = 1 if self.direction == UP else 0
        dir_right = 1 if self.direction == RIGHT else 0
        dir_down = 1 if self.direction == DOWN else 0
        dir_left = 1 if self.direction == LEFT else 0

    
        state = [
           
            danger_up, danger_right, danger_down, danger_left,
            
            wall_up, wall_right, wall_down, wall_left,
          
            dir_up, dir_right, dir_down, dir_left,
            
            food_dx / GRID_WIDTH, food_dy / GRID_HEIGHT
        ]

        return np.array(state, dtype=np.float32)

    def draw(self, surface):
      
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(surface, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))

      
        pygame.draw.rect(surface, WALL_COLOR, (0, 0, SCREEN_WIDTH, GRID_SIZE))  # 上墙
        pygame.draw.rect(surface, WALL_COLOR, (0, SCREEN_HEIGHT - GRID_SIZE, SCREEN_WIDTH, GRID_SIZE))  # 下墙
        pygame.draw.rect(surface, WALL_COLOR, (0, 0, GRID_SIZE, SCREEN_HEIGHT))  # 左墙
        pygame.draw.rect(surface, WALL_COLOR, (SCREEN_WIDTH - GRID_SIZE, 0, GRID_SIZE, SCREEN_HEIGHT))  # 右墙

 
        food_rect = pygame.Rect(self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, FOOD_COLOR, food_rect, border_radius=8)
        pygame.draw.rect(surface, (255, 150, 150), food_rect, 2, border_radius=8)


        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            if i == 0: 
                pygame.draw.rect(surface, SNAKE_HEAD, rect, border_radius=10)
                pygame.draw.rect(surface, (100, 220, 255), rect, 2, border_radius=10)

                eye_size = GRID_SIZE // 5
                if self.direction == RIGHT:
                    pygame.draw.circle(surface, (0, 0, 0), (rect.right - eye_size, rect.top + eye_size * 2), eye_size)
                    pygame.draw.circle(surface, (0, 0, 0), (rect.right - eye_size, rect.bottom - eye_size * 2),
                                       eye_size)
                elif self.direction == LEFT:
                    pygame.draw.circle(surface, (0, 0, 0), (rect.left + eye_size, rect.top + eye_size * 2), eye_size)
                    pygame.draw.circle(surface, (0, 0, 0), (rect.left + eye_size, rect.bottom - eye_size * 2), eye_size)
                elif self.direction == UP:
                    pygame.draw.circle(surface, (0, 0, 0), (rect.left + eye_size * 2, rect.top + eye_size), eye_size)
                    pygame.draw.circle(surface, (0, 0, 0), (rect.right - eye_size * 2, rect.top + eye_size), eye_size)
                elif self.direction == DOWN:
                    pygame.draw.circle(surface, (0, 0, 0), (rect.left + eye_size * 2, rect.bottom - eye_size), eye_size)
                    pygame.draw.circle(surface, (0, 0, 0), (rect.right - eye_size * 2, rect.bottom - eye_size),
                                       eye_size)
            else:  
                pygame.draw.rect(surface, SNAKE_BODY, rect, border_radius=6)
                pygame.draw.rect(surface, (60, 150, 220), rect, 1, border_radius=6)

        score_text = font_large.render(f"分数: {self.score}", True, TEXT_COLOR)
        surface.blit(score_text, (20, 20))



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob



class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

       
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)


        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

     
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

   
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)


        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if deterministic:
            mean, _ = self.policy_net(state)
            action = torch.tanh(mean)
        else:
            action, _ = self.policy_net.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=2048):
        if len(replay_buffer) < batch_size:
            return

       
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)


        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

       
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_net.sample(next_states)
            q1_next = self.target_q_net1(next_states, next_actions)
            q2_next = self.target_q_net2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * q_next

        q1 = self.q_net1(states, actions)
        q1_loss = F.mse_loss(q1, target_q)

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
       
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_norm=1.0)
        self.q_optimizer1.step()

        q2 = self.q_net2(states, actions)
        q2_loss = F.mse_loss(q2, target_q)

        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_norm=1.0)
        self.q_optimizer2.step()

        new_actions, log_probs = self.policy_net.sample(states)
        q1_new = self.q_net1(states, new_actions)
        q2_new = self.q_net2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()


        for param, target_param in zip(self.q_net1.parameters(), self.target_q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q_net2.parameters(), self.target_q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()

class ReplayBuffer:
    def __init__(self, capacity=2000000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

game = SnakeGame()
state_dim = len(game.get_state())
action_dim = 4  # 4个动作：上、右、下、左
agent = SACAgent(state_dim, action_dim)
replay_buffer = ReplayBuffer(capacity=4000000)

episodes = 1000
batch_size = 2048  # 增大批大小以利用GPU并行性
update_interval = 4
max_steps = 1000

scores = []
avg_scores = []
losses = []
episode_times = []
best_score = 0

train_button = Button(SCREEN_WIDTH - 200, SCREEN_HEIGHT - 80, 180, 50, "开始训练")
reset_button = Button(SCREEN_WIDTH - 200, SCREEN_HEIGHT - 150, 180, 50, "重置游戏")
speed_button = Button(SCREEN_WIDTH - 200, SCREEN_HEIGHT - 220, 180, 50, "加速模式")

running = True
training = True
high_speed = True
episode = 0
total_steps = 0
last_update_time = time.time()
last_save_time = time.time()

while running:
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if train_button.is_clicked(mouse_pos, event):
            training = not training
            train_button.text = "停止训练" if training else "开始训练"

        if reset_button.is_clicked(mouse_pos, event):
            game.reset()
            episode = 0
            scores.clear()
            avg_scores.clear()
            losses.clear()
            best_score = 0

        if speed_button.is_clicked(mouse_pos, event):
            high_speed = not high_speed
            speed_button.text = "正常速度" if high_speed else "加速模式"

  
    train_button.check_hover(mouse_pos)
    reset_button.check_hover(mouse_pos)
    speed_button.check_hover(mouse_pos)

    if not game.game_over:
      
        state = game.get_state()

        if training:
            action = agent.get_action(state)
            action_idx = np.argmax(action)
        else:
            
            action_idx = random.randint(0, 3)

        next_state, reward, done = game.step(action_idx)

        if training:
            replay_buffer.add(state, action, reward, next_state, done)
            total_steps += 1

         
            if total_steps % update_interval == 0:
                loss = agent.update(replay_buffer, batch_size)
                if loss is not None:
                    losses.append(loss)

 
        if done:
            scores.append(game.score)
            if len(scores) > 100:
                scores.pop(0)
            avg_score = np.mean(scores) if scores else 0
            avg_scores.append(avg_score)

            if game.score > best_score:
                best_score = game.score

            episode_times.append(time.time() - last_update_time)
            last_update_time = time.time()
            episode += 1

            if training and episode % 100 == 0:
                torch.save({
                    'q_net1_state_dict': agent.q_net1.state_dict(),
                    'q_net2_state_dict': agent.q_net2.state_dict(),
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.policy_optimizer.state_dict(),
                    'episode': episode,
                    'score': game.score,
                    'best_score': best_score
                }, f'snake_sac_model_ep{episode}_score{best_score}.pth')
                print(f"模型已保存: episode={episode}, best_score={best_score}")

            game.reset()
    else:
        game.reset()

    screen.fill(BACKGROUND)
    game.draw(screen)

    info_panel = pygame.Surface((360, 240), pygame.SRCALPHA)
    info_panel.fill(INFO_BG)
    screen.blit(info_panel, (SCREEN_WIDTH - 380, 20))

    
    if training:
        status_text = font_medium.render(f"训练中... 回合: {episode}", True, GREEN)
    else:
        status_text = font_medium.render(f"演示模式", True, RED)
    screen.blit(status_text, (SCREEN_WIDTH - 360, 30))

    score_text = font_small.render(f"当前分数: {game.score}", True, TEXT_COLOR)
    screen.blit(score_text, (SCREEN_WIDTH - 360, 70))

    if avg_scores:
        avg_text = font_small.render(f"平均分数: {avg_scores[-1]:.1f}", True, TEXT_COLOR)
        screen.blit(avg_text, (SCREEN_WIDTH - 360, 100))

  
    best_text = font_small.render(f"最佳分数: {best_score}", True, TEXT_COLOR)
    screen.blit(best_text, (SCREEN_WIDTH - 360, 130))

    buffer_text = font_small.render(f"经验缓冲区: {len(replay_buffer)}", True, TEXT_COLOR)
    screen.blit(buffer_text, (SCREEN_WIDTH - 360, 160))

    if episode_times:
        time_text = font_small.render(f"时间: {episode_times[-1]:.2f}秒", True, TEXT_COLOR)
        screen.blit(time_text, (SCREEN_WIDTH - 360, 190))

    if torch.cuda.is_available():
        gpu_text = font_small.render(f"GPU: {torch.cuda.get_device_name(0)}", True, GREEN)
        gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 2
        gpu_mem_text = font_small.render(f"显存: {gpu_mem:.1f} MB", True, GREEN)
    else:
        gpu_text = font_small.render("GPU: 不可用 (使用CPU)", True, RED)
        gpu_mem_text = font_small.render("", True, GREEN)

    screen.blit(gpu_text, (SCREEN_WIDTH - 360, 220))
    screen.blit(gpu_mem_text, (SCREEN_WIDTH - 360, 250))

    train_button.draw(screen)
    reset_button.draw(screen)
    speed_button.draw(screen)


    sac_text = font_medium.render("Soft Actor-Critic (SAC)", True, PURPLE)
    screen.blit(sac_text, (SCREEN_WIDTH // 2 - 180, 20))

    info_texts = [
        
        f"{device}",
        f"批大小: {batch_size}",
        f"更新间隔: {update_interval}步",
        
    ]

    for i, text in enumerate(info_texts):
        info_surf = font_small.render(text, True, (180, 180, 220))
        screen.blit(info_surf, (20, SCREEN_HEIGHT - 160 + i * 25))

    pygame.display.flip()

  
    if high_speed and training:
        clock.tick(0)  
    else:
        clock.tick(FPS)

pygame.quit()
