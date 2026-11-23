Preview and storyboards
=======================

``manim preview`` plays a scene in a window while you write it, and
``manim storyboard`` draws a contact sheet of its shots.  Both use the Vulkan
renderer.

Preview
-------

.. code-block:: console

    manim preview scene.py MyScene          # 480p15 by default
    manim preview -q m scene.py MyScene     # 720p30

The scene plays in real time.  Saving the file reloads it and shows the
current shot again; an exception is shown at the top of the window until the
next save.  A bar along the bottom shows the shots (one per ``play()`` or
``wait()`` call), the playhead, the current shot and the line of code that
made it.

=================  ===========================================================
Space              pause / play
Left / Right       previous / next shot
``,`` / ``.``      one frame back / forward
Home               start
``l``              loop the current shot
``r``              reload now
``s``              save the current frame as ``<file>_<time>s.png``
``[`` / ``]``      half / double speed
``g``              coordinate grid, with the cursor's scene coordinates
``e``              open the shot's line in an editor (``MANIM_EDITOR``,
                   default ``code -g {file}:{line}``)
click on the bar   seek
``q`` / Escape     quit
=================  ===========================================================

A scene runs forward: the state at a time depends on the plays and updaters
before it.  To show the scene from a time, the preview evaluates it again from
the start, drawing only the frames from that time on, so updaters see the same
history as in a render.  Frames already shown are kept, so stepping back while
paused is instant.

Storyboard
----------

.. code-block:: console

    manim storyboard scene.py MyScene -o board.png

writes one thumbnail per shot (its last frame) with its section, animations,
time range and line.  From Python:

.. code-block:: python

    from manim.preview import capture_storyboard, shot_thumbnails, storyboard_sheet

    board = capture_storyboard(MyScene)
    for shot in board.shots:
        print(shot.index, shot.label, shot.start, shot.end, shot.line)
    storyboard_sheet(board, shot_thumbnails(MyScene, board)).save("board.png")
