import random
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from base_env import BaseEnv

class WordleEnv(BaseEnv):
    def __init__(self, word_list: Optional[List[str]] = None, allowed_guesses: Optional[List[str]] = None, max_attempts: int = 6):
        """
        Initialize the Wordle environment.
        """
        if word_list is None:
            # Default word list
            self.word_list = ['apple', 'banjo', 'cabin', 'delta', 'eagle']
        else:
            self.word_list = word_list

        if allowed_guesses is None:
            self.allowed_guesses = self.word_list
        else:
            self.allowed_guesses = allowed_guesses

        self.max_attempts = max_attempts
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """
        Reset the game state to start a new game.
        """
        self.target_word = random.choice(self.word_list)
        self.guesses = []
        self.feedbacks = []
        self.attempt = 0
        self.done = False
        return self._get_observation()

    def step(self, action: str) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        """
        Take a step in the environment by making a guess.
        """
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        action = action.lower()

        if action not in self.allowed_guesses:
            reward = 0
            done = False
            info = {'valid': False, 'message': 'Invalid guess word.'}
            observation = self._get_observation()
            return observation, reward, done, info

        self.attempt += 1
        feedback = self._generate_feedback(action, self.target_word)
        self.guesses.append(action)
        self.feedbacks.append(feedback)

        if action == self.target_word:
            reward = 1
            self.done = True
            info = {'valid': True, 'message': 'Correct guess!'}
        elif self.attempt >= self.max_attempts:
            reward = 0
            self.done = True
            info = {'valid': True, 'message': 'Max attempts reached. Game over.'}
        else:
            reward = 0
            self.done = False
            info = {'valid': True, 'message': 'Incorrect guess.'}

        observation = self._get_observation()
        return observation, reward, self.done, info

    def render(self, mode: str = 'human') -> Optional[Image.Image]:
        """
        Render the current game state as an image.
        """
        img = self._create_image()
        if mode == 'human':
            img.show()
        else:
            return img

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass

    def _generate_feedback(self, guess: str, target: str) -> List[str]:
        """
        Generate feedback for a guess compared to the target word.
        """
        feedback = ['grey'] * 5
        target_letters = list(target)
        guess_letters = list(guess)

        # First pass: Check for correct positions (green)
        for i in range(5):
            if guess_letters[i] == target_letters[i]:
                feedback[i] = 'green'
                target_letters[i] = None  # Mark as used
                guess_letters[i] = None

        # Second pass: Check for correct letters in wrong positions (yellow)
        for i in range(5):
            if guess_letters[i] and guess_letters[i] in target_letters:
                feedback[i] = 'yellow'
                target_index = target_letters.index(guess_letters[i])
                target_letters[target_index] = None  # Mark as used

        return feedback

    def _get_observation(self) -> Dict[str, Any]:
        """
        Get the current observation of the game state.
        """
        return {
            'guesses': self.guesses,
            'feedbacks': self.feedbacks
        }

    def _create_image(self) -> Image.Image:
        """
        Create an image representing the current game state.
        """
        tile_size = 60
        grid_width = 5 * tile_size
        grid_height = self.max_attempts * tile_size
        img = Image.new('RGB', (grid_width, grid_height), color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except IOError:
            font = ImageFont.load_default(size=40)


        colors = {
            'green': (106, 170, 100),
            'yellow': (201, 180, 88),
            'grey': (120, 124, 126),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }

        for attempt in range(self.max_attempts):
            if attempt < len(self.guesses):
                guess = self.guesses[attempt]
                feedback = self.feedbacks[attempt]
            else:
                guess = ''
                feedback = ['white'] * 5

            for i in range(5):
                x0 = i * tile_size
                y0 = attempt * tile_size
                x1 = x0 + tile_size
                y1 = y0 + tile_size

                letter = guess[i].upper() if i < len(guess) else ''
                color = colors[feedback[i]] if i < len(feedback) else colors['white']

                draw.rectangle([x0, y0, x1, y1], fill=color, outline=colors['black'])

                text_x = x0 + tile_size / 4
                text_y = y0 + tile_size / 8
                draw.text((text_x, text_y), letter, fill=colors['white'], font=font)

        return img

if __name__ == "__main__":
    # Test code for the WordleEnv
    env = WordleEnv()
    observation = env.reset()
    done = False

    # For testing purposes, let's use a fixed target word to ensure consistent results
    env.target_word = 'apple'

    # Sample test guesses
    test_guesses = ['banjo', 'cabin', 'delta', 'eagle', 'apple']

    for guess in test_guesses:
        print(f"Guessing: {guess}")
        observation, reward, done, info = env.step(guess)
        img = env.render(mode="RGB")
        print(info['message'])
        print(f"obs:{observation}, reward:{reward}, done:{done}, info:{info}")
        # img.save(f"wordle_{guess}.png")
        if done:
            break

    if reward == 1:
        print("Test passed: Correctly guessed the word.")
    else:
        print(f"Test failed: Did not guess the word. The word was {env.target_word.upper()}")

    env.close()
