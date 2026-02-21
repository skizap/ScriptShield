import random
import time
import sys
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Global constants for configuration
DEFAULT_WIDTH = 80
DEFAULT_HEIGHT = 24
MIN_SPEED = 0.02
MAX_SPEED = 0.1
COLORS = {
    'green': '\033[32m',
    'light_green': '\033[92m',
    'white': '\033[97m',
    'reset': '\033[0m'
}

@dataclass
class DropMetrics:
    """Stores metrics for a single rain drop."""
    length: int
    speed: float
    characters_changed: int = 0

def timing_decorator(func):
    """A decorator to track how long a function takes to execute."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # Hidden logging logic that the obfuscator should mangle nicely
        _hidden_log = f"Exec {func.__name__}: {end - start:.4f}s"
        return result
    return wrapper

class TerminalContext:
    """Context manager for handling terminal state."""
    def __enter__(self):
        sys.stdout.write("\033[?25l")  # Hide cursor
        sys.stdout.write("\033[2J\033[H")  # Clear screen
        sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write("\033[?25h")  # Show cursor
        sys.stdout.write("\033[0m\n")  # Reset colors
        if exc_type is KeyboardInterrupt:
            print("\nMatrix connection terminated by user.")
            return True # Suppress the exception

class CharacterGenerator:
    """Generates characters for the matrix rain."""
    
    # Class variable
    ALLOWED_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+{}[]|:;<>,.?/~`"

    def __init__(self, mode: str = 'mixed'):
        self.mode = mode

    def get_random_char(self) -> str:
        """Returns a random character based on the current mode."""
        if self.mode == 'binary':
            return str(random.randint(0, 1))
        elif self.mode == 'numeric':
            return str(random.randint(0, 9))
        else:
            return random.choice(self.ALLOWED_CHARS)

    @staticmethod
    def get_color(is_head: bool = False, intensity: float = 1.0) -> str:
        """Static method to determine color."""
        if is_head:
            return COLORS['white']
        elif intensity > 0.8:
            return COLORS['light_green']
        return COLORS['green']

class RainDrop:
    """Represents a single falling line of text."""
    
    def __init__(self, col: int, max_height: int, char_gen: CharacterGenerator):
        self.col = col
        self.max_height = max_height
        self.char_gen = char_gen
        self.reset()
        
    def reset(self):
        self.row = random.randint(-self.max_height, 0)
        self.length = random.randint(5, int(self.max_height * 0.8))
        self.speed = random.uniform(MIN_SPEED, MAX_SPEED)
        self.chars: List[str] = []
        self.last_update = time.time()
        self.metrics = DropMetrics(length=self.length, speed=self.speed)
        
    def should_update(self, current_time: float) -> bool:
        return (current_time - self.last_update) >= self.speed

    def move(self):
        self.row += 1
        # Add new character at the head
        self.chars.insert(0, self.char_gen.get_random_char())
        
        # Randomly change some existing characters (Matrix glitch effect)
        for i in range(1, len(self.chars)):
            if random.random() < 0.1:
                self.chars[i] = self.char_gen.get_random_char()
                self.metrics.characters_changed += 1
                
        # Trim tail
        if len(self.chars) > self.length:
            self.chars.pop()
            
        if self.row - self.length > self.max_height:
            self.reset()

class MatrixSimulation:
    """Main simulation controller."""
    
    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.width = width
        self.height = height
        self.char_gen = CharacterGenerator(mode='mixed')
        self.drops = [RainDrop(c, height, self.char_gen) for c in range(width)]
        self.running = False
        
    def _get_terminal_size(self) -> Tuple[int, int]:
        """Try to get actual terminal size, fallback to defaults."""
        try:
            sz = os.get_terminal_size()
            return sz.columns, sz.lines
        except OSError:
            return DEFAULT_WIDTH, DEFAULT_HEIGHT

    def adjust_size(self):
        """Dynamically adjust to terminal resizing."""
        new_w, new_h = self._get_terminal_size()
        if new_w != self.width or new_h != self.height:
            self.width, self.height = new_w, new_h
            # Recreate drops using list comprehension
            self.drops = [RainDrop(c, self.height, self.char_gen) for c in range(self.width)]
            sys.stdout.write("\033[2J") # Clear screen on resize

    def draw_frame(self):
        """Draws the current state of all drops."""
        # Create an empty buffer using a dict comprehension
        buffer: Dict[Tuple[int, int], Tuple[str, str]] = {
            (r, c): (' ', COLORS['reset']) 
            for r in range(self.height) 
            for c in range(self.width)
        }
        
        # Fill buffer
        for drop in self.drops:
            for i, char in enumerate(drop.chars):
                r = drop.row - i
                if 0 <= r < self.height:
                    is_head = (i == 0)
                    intensity = 1.0 - (i / drop.length)
                    color = CharacterGenerator.get_color(is_head, intensity)
                    buffer[(r, drop.col)] = (char, color)

        # Render buffer
        sys.stdout.write("\033[H") # Move to top-left
        
        # Complex string manipulation to test string obfuscation
        output_lines = []
        for r in range(self.height):
            line_parts = []
            current_color = None
            
            for c in range(self.width):
                char, color = buffer[(r, c)]
                if color != current_color:
                    line_parts.append(color)
                    current_color = color
                line_parts.append(char)
                
            output_lines.append("".join(line_parts))
            
        sys.stdout.write("\n".join(output_lines))
        sys.stdout.flush()

    @timing_decorator
    def run(self, max_frames: Optional[int] = None):
        """Main loop."""
        self.running = True
        frame_count = 0
        
        with TerminalContext():
            while self.running:
                if max_frames is not None and frame_count >= max_frames:
                    break
                    
                current_time = time.time()
                
                # Check for resize occasionally
                if frame_count % 10 == 0:
                    self.adjust_size()
                
                # Update logic
                needs_draw = False
                for drop in self.drops:
                    if drop.should_update(current_time):
                        drop.move()
                        drop.last_update = current_time
                        needs_draw = True
                        
                if needs_draw:
                    self.draw_frame()
                    frame_count += 1
                    
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)

def run_complex_simulation():
    """Entry point with try-except-finally block."""
    sim = None
    try:
        sim = MatrixSimulation()
        sim.run()
    except Exception as e:
        print(f"Simulation crashed: {e}", file=sys.stderr)
    finally:
        if sim:
            print(f"\nTotal drops simulated: {len(sim.drops)}")

if __name__ == "__main__":
    run_complex_simulation()
