# Create vernier, crowding and uncrowding patches
import os, numpy, random, scipy.misc, matplotlib.pyplot as plt

# Create vernier patch
def createVernierPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir):

	patch = numpy.zeros((2*barHeight+offsetHeight, 2*barWidth+offsetWidth))
	patch[0                     :barHeight, 0                   :barWidth] = 254.0
	patch[barHeight+offsetHeight:         , barWidth+offsetWidth:        ] = 254.0

	if offsetDir:
		patch = numpy.fliplr(patch)

	return patch

# Create square patch
def createSquarePatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir):

	critDist  = (barHeight+offsetHeight)/2
	squareDim = max(2*barHeight+offsetHeight, 2*barWidth+offsetWidth)+2*critDist+2*barWidth
	patch     = numpy.zeros((squareDim, squareDim))
	patch[0:barWidth, :] = 254.0
	patch[-barWidth:, :] = 254.0
	patch[:, 0:barWidth] = 254.0
	patch[:, -barWidth:] = 254.0

	return patch

# Create crowded patch
def createCrowdedPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir):

	critDist        = (barHeight+offsetHeight)/2
	patch           = createSquarePatch( barHeight, barWidth, offsetHeight, offsetWidth, offsetDir)
	vernierPatch    = createVernierPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir)
	firstVernierRow = critDist+barWidth
	firstVernierCol = patch.shape[0]/2 - (barWidth+offsetWidth/2)

	patch[firstVernierRow:firstVernierRow+2*barHeight+offsetHeight, firstVernierCol:firstVernierCol+2*barWidth+offsetWidth] = vernierPatch

	return patch

# Create uncrowded patch (7 squares)
def createUncrowdedPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir):

	critDist     = (barHeight+offsetHeight)/2
	squarePatch  = createSquarePatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir)
	crowdedPatch = createCrowdedPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir)
	oneSquareDim = squarePatch.shape[0]
	patch        = numpy.zeros((oneSquareDim, 7*oneSquareDim + 6*critDist))

	for n in range(7):
		firstCol = n*(oneSquareDim+critDist)
		if n == 3:
			patch[:, firstCol:firstCol+oneSquareDim] = crowdedPatch
		else:
			patch[:, firstCol:firstCol+oneSquareDim] = squarePatch

	return patch

# Main function
def createPatches(nSamples, stimType):

	if not os.path.exists(stimType):
		os.mkdir(stimType)

	barHeightRange    = range(5, 20)
	barWidthRange     = range(2,  5)
	offsetHeightRange = range(0,  5)
	offsetWidthRange  = range(2,  7)
	names             = ['L', 'R']

	for offset in (0,1):
		name = stimType+'/'+names[offset]
		n    = 0
		while n < nSamples:

			bH = random.choice(barHeightRange)
			bW = random.choice(barWidthRange)
			oH = random.choice(offsetHeightRange)
			oW = random.choice(offsetWidthRange)

			if stimType   == 'vernier':
				thisPatch = numpy.pad(createVernierPatch(  bH, bW, oH, oW, offset), 10, mode='constant')
			elif stimType == 'crowded':
				thisPatch = numpy.pad(createCrowdedPatch(  bH, bW, oH, oW, offset), 10, mode='constant')
			elif stimType == 'uncrowded':
				thisPatch = numpy.pad(createUncrowdedPatch(bH, bW, oH, oW, offset), 10, mode='constant')
			else:
				raise Exception('Unknown stimulus type.')

			if thisPatch.shape[0] > 227 or thisPatch.shape[1] > 227:
				pass
			else:
				n += 1
				thisName = name+str(n)
				# numpy.save(thisName, numpy.array([thisPatch, thisPatch, thisPatch]))
				scipy.misc.imsave(thisName+'.png', thisPatch) # numpy.array([thisPatch, thisPatch, thisPatch]))

# Example main call
#createPatches(10, 'vernier')
#createPatches(10, 'crowded')
#createPatches(10, 'uncrowded')

# plt.figure()
# plt.imshow(thisPatch, interpolation='nearest')
# # ax = plt.gca()
# # ax.set_xticks(numpy.arange(0, thisPatch.shape[1], 1));
# # ax.set_yticks(numpy.arange(0, thisPatch.shape[0], 1));
# # ax.set_xticklabels(numpy.arange(1, thisPatch.shape[0]+1, 1));
# # ax.set_yticklabels(numpy.arange(1, thisPatch.shape[0]+1, 1));
# # ax.set_xticks(numpy.arange(-.5, thisPatch.shape[1], 1), minor=True);
# # ax.set_yticks(numpy.arange(-.5, thisPatch.shape[1], 1), minor=True);
# # ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
# plt.show()