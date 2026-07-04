import argparse
import random
import types
import numpy as np
import math

# Apply NumPy 2.0 backward compatibility monkey-patch for PsychoPy 2023
if not hasattr(np, 'alltrue'): np.alltrue = np.all
if not hasattr(np, 'sometrue'): np.sometrue = np.any
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

from psychopy import visual, core, event
import pylsl

matrixChars = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_']
]

DIM_COLOR = '#262626' # Dimmed white
FLASH_COLOR = '#00FF00'
FIXATION_COLOR = '#FF0000'
BG_COLOR = '#121212'
READY_COLOR = '#555555' # Faint white — used between letters so user can find next target

def get_char_pos(target_char):
    for r in range(6):
        for c in range(6):
            if matrixChars[r][c] == target_char:
                return r, c
    return None

def generate_flash_sequence(mode, target_char=None):
    """Generate a shuffled list of flash-group IDs for one block.

    mode == 1: Row-Column Paradigm (RCP).
        12 events per block: 6 rows (r0-r5) + 6 cols (c0-c5).
        Each char is in exactly 1 row + 1 col -> 2-membership, target
        rate 2/12 ≈ 16.7 %.

    mode == 2: Checkerboard-style Paradigm (CBP).
        12 events per block: 6 stride-diagonal groups (g0-g5) + 6
        rows (r0-r5). The stride groups come from ``(r*2 + k) % 6``,
        which scatters the 6 flashed cells across rows and 3 distinct
        columns — the spatial-adjacency-breaking property that
        motivates CBP over RCP. Each char at (r, c) is in exactly
        1 stride group (k = (c - 2r) mod 6) + 1 row (r), giving
        2-membership with unique (stride, row) signature per char.
        Target rate 2/12 ≈ 16.7 %.

        Historical note:
          - The originally-shipped CBP used only the 6 stride groups
            (g0-g5), giving each char membership 1 → information-
            theoretically incapable of disambiguating 1-of-36 →
            accumulator plateaued at ~14 % accuracy.
          - An intermediate fix used stride+cols, but chars at
            (r, c) and (r+3, c) shared the same (stride, col)
            signature, capping accuracy at ~50 %.
          - Current design uses stride+rows: all 36 signatures
            unique (verified in tests), measured ≥ 93 % accuracy at
            6 blocks on the held-out LDA p-distribution.
    """
    target_pos = get_char_pos(target_char) if target_char else None

    if mode == 1:
        items = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5',
                 'c0', 'c1', 'c2', 'c3', 'c4', 'c5']
    elif mode == 2:
        items = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5',
                 'r0', 'r1', 'r2', 'r3', 'r4', 'r5']
    else:
        raise ValueError(f"unknown mode={mode!r}; expected 1 (RCP) or 2 (CBP)")

    valid = False
    seq = []
    attempts = 0
    while not valid and attempts < 1000:
        attempts += 1
        seq = items[:]
        random.shuffle(seq)
        valid = True
        for i in range(len(seq) - 1):
            curr = seq[i]
            nxt = seq[i+1]

            # Constraint 1: Adjacency avoid — same axis, adjacent index.
            if curr[0] == nxt[0]:
                if abs(int(curr[1]) - int(nxt[1])) == 1:
                    valid = False
                    break

            # Constraint 2: don't flash two target-containing groups back-to-back.
            if target_pos:
                r, c = target_pos
                # Target lives in row r, col c, and in stride group k where
                # (r*2 + k) % 6 == c  ->  k = (c - 2*r) mod 6.
                target_stride_k = (c - 2 * r) % 6
                target_labels = {f'r{r}', f'c{c}', f'g{target_stride_k}'}
                if curr in target_labels and nxt in target_labels:
                    valid = False
                    break
    return seq


def main():
    from psychopy import gui
    
    # Updated Project 2 Config Dialog
    exp_info = {
        '01. System Mode': ['Project 2: LLM Speller', 'Standalone Training'],
        '02. Keyboard Mode': ['2: Checkerboard (CBP)', '1: Row-Column (RCP)'],
        '03. LLM Trigger Conf (0-1)': '0.8',
        '04. SSVEP Timeout (s)': '15',
        '05. Target Word (Training Only)': 'NEUROTECH',
        '06. Inference Trials': '5',
        '07. Blocks (Flashes)': '10',
        '08. Monitor FPS': '60'
    }
    
    dlg = gui.DlgFromDict(dictionary=exp_info, sortKeys=False, title="NeuroTech ASU - Project 2 Config")
    if not dlg.OK:
        core.quit()
        
    args = types.SimpleNamespace()
    
    try:
        # Map human-readable names to internal logic
        args.inference = (exp_info['01. System Mode'] == 'Project 2: LLM Speller')
        args.mode = 2 if 'Checkerboard' in exp_info['02. Keyboard Mode'] else 1
        args.llm_threshold = float(exp_info['03. LLM Trigger Conf (0-1)'])
        args.ssvep_timeout = float(exp_info['04. SSVEP Timeout (s)'])
        args.word = exp_info['05. Target Word (Training Only)']
        args.trials = int(exp_info['06. Inference Trials'])
        args.blocks = int(exp_info['07. Blocks (Flashes)'])
        args.fps = int(exp_info['08. Monitor FPS'])
    except ValueError:
        dlg2 = gui.Dlg(title="Invalid Input")
        dlg2.addText("Numerical fields received invalid input. Check Threshold or Timeout.")
        dlg2.show()
        core.quit()

    # Create the LSL Marker Outlet
    print("Initializing LSL Marker Outlet...")
    info = pylsl.StreamInfo('Speller_Markers', 'Markers', 1, 0, 'string', 'psychopy_v1')
    outlet = pylsl.StreamOutlet(info)

    # Initialize Window explicitly requesting FullScreen and locking V-Sync
    # This must happen BEFORE subprocess creation, otherwise Pyglet loses foreground focus 
    # and gets permanently stuck calculating the monitor refresh rate!
    win = visual.Window(size=[1280, 720], fullscr=True, allowGUI=False, 
                        color=BG_COLOR, units='norm', waitBlanking=True, checkTiming=False)

    # --- AUTO-LAUNCH BACKEND ---
    import subprocess
    import sys
    import os
    
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(os.path.dirname(ui_dir), "backend")
    backend_script = os.path.join(backend_dir, "realtime_inference.py" if args.inference else "data_collection.py")
    
    # Resolve the real Python interpreter — PsychoPy Runner sets sys.executable
    # to its own launcher (e.g., psychopy.exe), not the actual python.exe.
    # We need the raw interpreter to launch backend scripts as subprocesses.
    _exe_dir = os.path.dirname(sys.executable)
    _python_exe = os.path.join(_exe_dir, "python.exe")
    if not os.path.isfile(_python_exe):
        _python_exe = sys.executable  # Fallback
    
    print(f"Auto-launching Backend: {backend_script}")
    try:
        # Launch backend. Path/dependency fixes are handled inside the backend scripts.
        subprocess.Popen([_python_exe, backend_script])
        
        # Adaptive handshake: poll for backend's LSL stream instead of hardcoded wait.
        # For inference mode, wait until Speller_Decoded stream appears (model trained).
        # For training mode, wait until data_collection.py has resolved its streams.
        if args.inference:
            print("Waiting for backend to train models and publish Speller_Decoded stream...")
            max_wait = 30  # seconds — generous ceiling for model training
            for i in range(max_wait):
                handshake = pylsl.resolve_byprop('name', 'Speller_Decoded', 1, 1.0)
                if handshake:
                    print(f"Backend ready! Handshake completed in {i+1} seconds.")
                    break
                print(f"  Still waiting... ({i+1}/{max_wait}s)")
            else:
                print(f"WARNING: Backend did not publish Speller_Decoded within {max_wait}s. Proceeding anyway.")
        else:
            # Training mode: backend just needs to resolve LSL streams, which is fast
            print("Waiting 3 seconds for data collection backend to initialize...")
            core.wait(3.0)
    except Exception as e:
        print(f"Failed to auto-start backend: {e}")
                        
    # Setup grid spacing
    grid_stims = []
    text_size_base = 0.15
    text_size_pop = 0.25
    x_positions = [-0.6, -0.36, -0.12, 0.12, 0.36, 0.6]
    y_positions = [0.6, 0.36, 0.12, -0.12, -0.36, -0.6] 

    for r in range(6):
        row_stims = []
        for c in range(6):
            stim = visual.TextStim(win, text=matrixChars[r][c], 
                                   pos=(x_positions[c], y_positions[r]), 
                                   color=DIM_COLOR, height=text_size_base, bold=True)
            row_stims.append(stim)
        grid_stims.append(row_stims)

    # UI labels
    instruction = visual.TextStim(win, text="Initializing...", pos=(0, 0.85), color=FLASH_COLOR, height=0.08)
    typed_text = visual.TextStim(win, text="", pos=(0, -0.85), color='#FFFFFF', height=0.1)
    context_label = visual.TextStim(win, text="", pos=(0, 0.95), color='#00FFFF', height=0.05) # Cyan context word
    llm_response_text = visual.TextStim(win, text="", pos=(0, 0.2), color='#FFFFFF', height=0.08, wrapWidth=1.5)
    
    # Pre-compute frames for exact 150ms timings (300ms ISI total)
    # 300ms ISI eliminates ERP temporal overlap entirely.
    # The P300 response lasts 200-400ms; at 300ms ISI each ERP fully resolves
    # before the next flash, maximizing single-trial discriminability.
    # E.g., 60fps -> 9 frames is 150ms
    frames_flash_on = int(round((150.0 / 1000.0) * args.fps))
    frames_flash_off = int(round((150.0 / 1000.0) * args.fps))
    if frames_flash_on < 1: frames_flash_on = 1
    if frames_flash_off < 1: frames_flash_off = 1

    def draw_all():
        if current_state in ["CONTEXT_P300", "MAIN_SPELLER"]:
            for row in grid_stims:
                for stim in row:
                    stim.draw()
        instruction.draw()
        typed_text.draw()
        context_label.draw()

    def exec_flash(group_id, target_char):
        chars_flashed = []
        stims_to_pop = []
        
        # Determine group
        if group_id.startswith('r'):
            r = int(group_id[1])
            for c in range(6):
                stims_to_pop.append(grid_stims[r][c])
                chars_flashed.append(matrixChars[r][c])
        elif group_id.startswith('c'):
            c = int(group_id[1])
            for r in range(6):
                stims_to_pop.append(grid_stims[r][c])
                chars_flashed.append(matrixChars[r][c])
        elif group_id.startswith('g'):
            # CBP stride-diagonal group: 6 cells scattered across rows,
            # never two adjacent rows in the same column.
            k = int(group_id[1])
            for r in range(6):
                c = (r * 2 + k) % 6
                stims_to_pop.append(grid_stims[r][c])
                chars_flashed.append(matrixChars[r][c])
                
        tag = 1 if target_char and target_char in chars_flashed else 0
        if target_char:
            marker_str = f"FLASH_{tag}_{''.join(chars_flashed)}"
        else:
            marker_str = f"FLASH_GROUP_{''.join(chars_flashed)}"

        # POP ON State
        for stim in stims_to_pop:
            stim.color = FLASH_COLOR
            stim.height = text_size_pop
        
        draw_all()
        # Flip #1 (0ms onset)
        win.flip()
        # Push marker immediately using native LSL clock to guarantee exact Unicorn sync
        outlet.push_sample([marker_str], pylsl.local_clock())
        
        # Hold ON state for duration (already completed 1 frame via the flip above)
        for _ in range(frames_flash_on - 1):
            draw_all()
            win.flip()
            
        # POP OFF State
        for stim in stims_to_pop:
            stim.color = DIM_COLOR
            stim.height = text_size_base
            
        draw_all()
        win.flip()
        
        # Hold OFF state
        for _ in range(frames_flash_off - 1):
            draw_all()
            win.flip()

    current_spelled = ""
    current_context = ""
    current_state = "CONTEXT_SSVEP"
    state_start_time = core.getTime()
    autocomplete_words = []
    ssvep_freqs_context = [10.0, 15.0]
    ssvep_freqs_auto = [8.57, 10.0, 12.0, 15.0]

    if args.inference:
        dec_inlet = None
        for i in range(40):
            streams = pylsl.resolve_byprop('name', 'Speller_Decoded', 1, 1.0)
            if streams:
                dec_inlet = pylsl.StreamInlet(streams[0])
                break
            core.wait(1.0)
            
        if not dec_inlet: core.quit()

        outlet.push_sample(["SSVEP_START"], pylsl.local_clock())
        outlet.push_sample([f"SET_THRESHOLD:{args.llm_threshold}"], pylsl.local_clock())
        
        while True:
            if event.getKeys(keyList=['escape']):
                outlet.push_sample(["SESSION_END"], pylsl.local_clock())
                break
            t = core.getTime()

            if current_state == "CONTEXT_SSVEP":
                instruction.setText("SSVEP: Choose Context Mode")
                labels = ["Enter Context", "No Context"]
                for i, (txt, freq) in enumerate(zip(labels, ssvep_freqs_context)):
                    x = -600 if i == 0 else 600 # Maximum spacing
                    box = visual.Rect(win, width=350, height=350, pos=(x, 0), 
                                      fillColor=[1, 1, 1], units='pix')
                    lbl = visual.TextStim(win, text=txt, pos=(x, 0), 
                                          color=[-1, -1, -1], height=70, bold=True, units='pix', wrapWidth=320)
                    if math.sin(2 * math.pi * freq * t) > 0:
                        box.draw()
                        lbl.draw()
                instruction.draw()
                win.flip()
                marker, _ = dec_inlet.pull_sample(timeout=0.0)
                if marker and marker[0].startswith("SSVEP_DECODED_"):
                    f = float(marker[0].replace("SSVEP_DECODED_", ""))
                    if f == 10.0:
                        current_state = "CONTEXT_P300"
                        instruction.setText("P300: Spell Context Word...")
                    else:
                        current_state = "MAIN_SPELLER"
                        instruction.setText("P300: Spell Word (No Context)")
                    outlet.push_sample(["SSVEP_STOP"], pylsl.local_clock())
                    state_start_time = core.getTime()

            elif current_state in ["CONTEXT_P300", "MAIN_SPELLER"]:
                outlet.push_sample(["SESSION_START"], pylsl.local_clock())
                for _ in range(int(3.0 * args.fps)):
                    draw_all()
                    win.flip()
                char_found = None
                for b in range(args.blocks):
                    if char_found: break
                    seq = generate_flash_sequence(args.mode, None)
                    for group in seq:
                        exec_flash(group, None)
                        marker, _ = dec_inlet.pull_sample(timeout=0.0)
                        if marker:
                            if marker[0].startswith("DECODED_"):
                                char_found = marker[0].replace("DECODED_", "")
                                break
                            elif marker[0].startswith("SSVEP_PREDICTIONS:"):
                                autocomplete_words = marker[0].replace("SSVEP_PREDICTIONS:", "").split(",")
                                current_state = "AUTOCOMPLETE_SSVEP"
                                outlet.push_sample(["SSVEP_START"], pylsl.local_clock())
                                state_start_time = core.getTime()
                                break
                            elif marker[0].startswith("LLM_RESPONSE:"):
                                response_content = marker[0].replace("LLM_RESPONSE:", "")
                                llm_response_text.setText(response_content)
                                current_state = "LLM_RESPONSE_SCREEN"
                                outlet.push_sample(["SSVEP_START"], pylsl.local_clock())
                                state_start_time = core.getTime()
                                break
                    if current_state == "AUTOCOMPLETE_SSVEP" or current_state == "LLM_RESPONSE_SCREEN": break
                    if not char_found: outlet.push_sample(["EVALUATE"], pylsl.local_clock())
                if current_state == "AUTOCOMPLETE_SSVEP" or current_state == "LLM_RESPONSE_SCREEN": continue
                if not char_found:
                    outlet.push_sample(["TRIAL_END"], pylsl.local_clock())
                    for _ in range(int(3.0 * args.fps)):
                        draw_all(); win.flip()
                        marker, _ = dec_inlet.pull_sample(timeout=0.0)
                        if marker and marker[0].startswith("DECODED_"):
                            char_found = marker[0].replace("DECODED_", "")
                            break
                if char_found:
                    if current_state == "CONTEXT_P300":
                        if char_found == "_":
                            current_state = "MAIN_SPELLER"
                            # The user requested the context word disappears from the screen
                            context_label.setText("") 
                            instruction.setText("P300: Spell Word")
                            outlet.push_sample([f"SET_CONTEXT:{current_context}"], pylsl.local_clock())
                        elif char_found == "8": 
                            current_context = current_context[:-1]
                        elif char_found == "9":
                            pass # 9 is submit, ignore in context mode
                        else: 
                            current_context += char_found
                            
                        if current_state == "CONTEXT_P300":
                            instruction.setText(f"Context: {current_context}")
                    else:
                        if char_found == "8": 
                            current_spelled = current_spelled[:-1]
                        elif char_found == "_": 
                            current_spelled += " "
                        elif char_found == "9": 
                            current_spelled = "" # Visually wipe the spelling line on submit
                        else: 
                            current_spelled += char_found
                        typed_text.setText(current_spelled)
                for row in grid_stims:
                    for stim in row: stim.color = READY_COLOR
                for _ in range(int(3.0 * args.fps)):
                    draw_all(); win.flip()
                for row in grid_stims:
                    for stim in row: stim.color = DIM_COLOR

            elif current_state == "AUTOCOMPLETE_SSVEP":
                # 4-point distributed layout mapped to the extreme corners using normalized units
                positions = [(-0.85, 0.75), (0.85, 0.75), (-0.85, -0.75), (0.85, -0.75)]
                for i, (word, freq) in enumerate(zip(autocomplete_words, ssvep_freqs_auto)):
                    pos = positions[i]
                    # Dotted/Checkerboard pattern to reduce glare (sf controls the number of dots)
                    box = visual.GratingStim(win, tex='sqrXsqr', size=(0.35, 0.35), pos=pos, 
                                             sf=15.0, color=[1, 1, 1], contrast=0.6, units='norm')
                    # High contrast dark plate behind the text
                    bg = visual.Rect(win, width=0.35, height=0.15, pos=pos, fillColor=[-0.9, -0.9, -0.9], units='norm')
                    lbl = visual.TextStim(win, text=word, pos=pos, 
                                          color=[1, 1, 1], height=0.12, bold=True, units='norm', wrapWidth=0.35)
                    if math.sin(2 * math.pi * freq * t) > 0:
                        box.draw()
                        bg.draw()
                        lbl.draw()
                win.flip()
                if core.getTime() - state_start_time > args.ssvep_timeout:
                    current_state = "MAIN_SPELLER"
                    outlet.push_sample(["SSVEP_TIMEOUT"], pylsl.local_clock())
                    state_start_time = core.getTime()
                marker, _ = dec_inlet.pull_sample(timeout=0.0)
                if marker and marker[0].startswith("SSVEP_DECODED_"):
                    f = float(marker[0].replace("SSVEP_DECODED_", ""))
                    try:
                        idx = ssvep_freqs_auto.index(f)
                        selected_word = autocomplete_words[idx]
                        parts = current_spelled.strip().split()
                        if parts: parts[-1] = selected_word
                        else: parts = [selected_word]
                        current_spelled = " ".join(parts) + " "
                        typed_text.setText(current_spelled)
                        current_state = "MAIN_SPELLER"
                        outlet.push_sample([f"WORD_SELECTED:{selected_word}"], pylsl.local_clock())
                        state_start_time = core.getTime()
                    except ValueError: pass

            elif current_state == "LLM_RESPONSE_SCREEN":
                llm_response_text.draw()
                
                elapsed_time = core.getTime() - state_start_time
                
                if elapsed_time > 5.0:
                    # Place Continue in the bottom right corner
                    pos = (0.85, -0.75)
                    box = visual.GratingStim(win, tex='sqrXsqr', size=(0.35, 0.3), pos=pos, 
                                             sf=15.0, color=[1, 1, 1], contrast=0.6, units='norm')
                    bg = visual.Rect(win, width=0.35, height=0.15, pos=pos, fillColor=[-0.9, -0.9, -0.9], units='norm')
                    lbl = visual.TextStim(win, text="Continue", pos=pos, 
                                          color=[1, 1, 1], height=0.12, bold=True, units='norm')
                    
                    if math.sin(2 * math.pi * 15.0 * t) > 0:
                        box.draw()
                        bg.draw()
                        lbl.draw()
                
                win.flip()
                
                marker, _ = dec_inlet.pull_sample(timeout=0.0)
                if marker and marker[0].startswith("SSVEP_DECODED_"):
                    if elapsed_time > 5.0:
                        f = float(marker[0].replace("SSVEP_DECODED_", ""))
                        if f == 15.0:
                            current_spelled = ""
                            typed_text.setText("")
                            current_state = "MAIN_SPELLER"
                            outlet.push_sample(["RESPONSE_ACK"], pylsl.local_clock())
                            state_start_time = core.getTime()
    else:
        for char in args.word:
            pos = get_char_pos(char)
            if pos is None: continue
            r, c = pos
            target_stim = grid_stims[r][c]
            instruction.setText(f"Focus on '{char}'...")
            target_stim.color = FIXATION_COLOR
            target_stim.height = text_size_pop
            for _ in range(int(3.0 * args.fps)):
                draw_all(); win.flip()
            target_stim.color = DIM_COLOR
            target_stim.height = text_size_base
            instruction.setText("Flashing...")
            for b in range(args.blocks):
                seq = generate_flash_sequence(args.mode, char)
                for group in seq:
                    exec_flash(group, char)
                    if event.getKeys(keyList=['escape']):
                        win.close(); core.quit()
            current_spelled += char
            typed_text.setText(current_spelled)
            for row in grid_stims:
                for stim in row: stim.color = READY_COLOR
            for _ in range(int(3.0 * args.fps)):
                draw_all(); win.flip()
            for row in grid_stims:
                for stim in row: stim.color = DIM_COLOR
            if char == '_':
                for countdown in range(30, 0, -1):
                    instruction.setText(f"Word complete! REST: {countdown}s")
                    for _ in range(int(1.0 * args.fps)):
                        draw_all(); win.flip()
                        if event.getKeys(keyList=['escape']):
                            win.close(); core.quit()
        outlet.push_sample(["SESSION_END"], pylsl.local_clock())

    win.close()
    core.quit()

if __name__ == '__main__':
    main()
