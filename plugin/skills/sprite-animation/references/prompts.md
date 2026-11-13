# Proven prompts

Templates that worked for a realistic painted character. Replace the [brackets]; always name the clothing in
detail and the exact chroma color. "Right" means the right side of the picture. The talking and gesture prompts
are derived from the notes but not yet tested word for word.

## Views (`edit_image`)

**BR from FR** (`images` = [fr.png]):

> Turn the [man] from image 1 around so that he is seen three-quarters from behind: his back is toward the viewer
> and his body, hips, shoulders and head are turned 45 degrees toward the right side of the picture; we see the
> back of his head and a little of his right cheek, not his face. He stands relaxed, heels toward the viewer, toes
> of both shoes pointing slightly outward, feet hip-width apart. Keep exactly the same person, the same clothing
> ([clothing]), the same size and position in the picture, the same painted style, and the same solid flat pure
> green (#00FF00) background.

**B from BR** (`images` = [br.png]):

> Turn the [man] from image 1 a little further so that he is seen straight from behind: his back faces the viewer
> directly. The pose is perfectly symmetrical: shoulders level, both arms hanging the same way at his sides, heels
> toward the viewer, the toes of both shoes pointing equally slightly outward, feet hip-width apart, centered in
> the picture. Keep exactly the same person, the same hair, the same clothing ([clothing, incl. back details such
> as a center back vent]), the same size, the same painted style, and the same solid flat pure green (#00FF00)
> background.

**F from FR** (middle keyframe for the turn FR→FL): as BR, but "turned to face the viewer directly, symmetrical
pose".

**Outfit variant** (`images` = [original view, outfit view already chosen]):

> Dress the [man] from image 1 in exactly the clothes of the [man] from image 2 ([new clothing]). Change only the
> clothes: keep the pose, the view, the size and position of image 1, the same painted style and the same solid
> flat pure green (#00FF00) background.

## Walk with start and stop (`generate_video`, [stand, stand], 5 s)

> The [man] first stands still and relaxed for a moment, then starts walking in place, as on a treadmill, with a
> steady calm walking cycle, legs and arms swinging alternately. After several steps he slows down and comes to a
> stop, ending standing still and relaxed exactly as at the beginning. He stays in the center of the frame at the
> same size and keeps facing three-quarters toward the right of the picture the whole time. Static camera, solid
> flat pure green (#00FF00) background, no shadow, no other objects.

For back views replace the facing sentence: "keeps his back turned three-quarters toward the right of the picture
the whole time and never turns his back straight toward the viewer".

## Walk cycle only ([stand, stand], 3 s)

> The [character] walks in place with a steady walking cycle, as on a treadmill, legs and arms swinging
> alternately; it stays in the center of the frame and keeps facing [front-right]. Static camera, three-quarter
> view, solid flat pure [green (#00FF00)] background, no shadow, no other objects.

## Turn on the spot ([from, to], 2 s; FR→FL: [fr, f, fl])

> The [man] turns on the spot from [facing three-quarters toward the right] to [being seen from behind]. He turns
> away from the viewer toward the right. His feet stay on the same spot, with one or two small steps. Static
> camera, same size, solid flat pure green (#00FF00) background, no shadow.

Via the front: "He turns to face the viewer directly and then further to the left. He never shows us his back."

## Idle ([stand, stand], 4 s, drop the last frame)

> The [man] stands almost completely motionless, like someone waiting patiently: the only movement is a very
> slight, slow rise and fall of the shoulders and chest as he breathes. His head stays completely still, his arms
> hang relaxed. He does not walk, turn, nod, look around or gesture. Static camera, solid flat pure green
> (#00FF00) background, no shadow.

## Talking ([stand, stand], 4 s, drop the last frame; only views with a face)

> The [man] stands still and talks calmly to someone in front of him: his mouth moves with the words, slight
> natural head movements, small hand movements at waist height. He does not walk or turn. Static camera, solid
> flat pure green (#00FF00) background, no shadow.

## Gesture ([stand, stand], 3 s)

> The [man] reaches with his empty right hand [up to a shelf above his head / to something at hip height / down
> to the floor], takes something invisible and returns to standing relaxed exactly as at the beginning. His hand
> stays empty. His feet stay on the same spot. Static camera, solid flat pure green (#00FF00) background, no
> shadow.
