// here is the whole process of this model:

// step 1. detech the boundary of each face.
// 1. download the model into ./models


// 2. loading the model needed. Here the SSD was loaded.
// await faceapi.loadSsdMobilenetv1Model('/models')
// accordingly for the other models:
// await faceapi.loadTinyFaceDetectorModel('/models')
// await faceapi.loadMtcnnModel('/models')
// await faceapi.loadFaceLandmarkModel('/models')
// await faceapi.loadFaceLandmarkTinyModel('/models')
// await faceapi.loadFaceRecognitionModel('/models')
// await faceapi.loadFaceExpressionModel('/models')

const MODEL_URL = '/models'

await faceapi.loadSsdMobilenetv1Model(MODEL_URL)
await faceapi.loadFaceLandmarkModel(MODEL_URL)
await faceapi.loadFaceRecognitionModel(MODEL_URL)

// 3. To detect all face’s bounding boxes of an input image we simply say:
const input = document.getElementById('myImage')
let fullFaceDescriptions = await faceapi.detectAllFaces(input).withFaceLandmarks().withFaceDescriptors()

// 4. get the description(bound) of target picture
fullFaceDescriptions = fullFaceDescriptions.map(fd => fd.forSize(width, height))

// 5. display and visualize the result.

const detectionsArray = fullFaceDescriptions.map(fd => fd.detection)
faceapi.drawDetection(canvas, detectionsArray, { withScore: true })


// 6. display the landmarks.

const landmarksArray = fullFaceDescriptions.map(fd => fd.landmarks)
faceapi.drawLandmarks(canvas, landmarksArray, { drawLines: true })


// step 2. Face recognition


// 7. using faceapi.fetchImage for face recognition
const labels = ['sheldon' 'raj', 'leonard', 'howard']

const labeledFaceDescriptors = await Promise.all(
  labels.map(async label => {
    // fetch image data from urls and convert blob to HTMLImage element
    const imgUrl = `${label}.png`
    const img = await faceapi.fetchImage(imgUrl)

    // detect the face with the highest score in the image and compute it's landmarks and face descriptor
    const fullFaceDescription = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()

    if (!fullFaceDescription) {
      throw new Error(`no faces detected for ${label}`)
    }

    const faceDescriptors = [fullFaceDescription.descriptor]
    return new faceapi.LabeledFaceDescriptors(label, faceDescriptors)
  })
)


// 8. determine the threshold.

// 0.6 is a good distance threshold value to judge
// whether the descriptors match or not
const maxDescriptorDistance = 0.6
const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance)

const results = fullFaceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor))


// 9. drawing the boundary for each face.

const boxesWithText = results.map((bestMatch, i) => {
  const box = fullFaceDescriptions[i].detection.box
  const text = bestMatch.toString()
  const boxWithText = new faceapi.BoxWithText(box, text)
  return boxWithText
})

faceapi.drawDetection(canvas, boxesWithText)
