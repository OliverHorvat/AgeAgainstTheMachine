package hr.ferit.oliverhorvat.ageagainstthemachine

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import hr.ferit.oliverhorvat.ageagainstthemachine.ml.BestModel
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfRect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private val cameraRequestCode = 1
    private val galleryRequestCode = 2
    private var currentPhotoPath = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (OpenCVLoader.initDebug()){
            Log.d("LOADED","success")
        }

        else {
            Log.d("LOADED","err")
        }

        val cameraButton = findViewById<Button>(R.id.cameraButton)
        val galleryButton = findViewById<Button>(R.id.galleryButton)

        val requestCameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {

                val fileName = "photo"
                val storageDirectory = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
                val imageFile = File.createTempFile(fileName, ".jpg", storageDirectory)
                currentPhotoPath = imageFile.absolutePath
                val imageUri = FileProvider.getUriForFile(this, "hr.ferit.oliverhorvat.ageagainstthemachine.fileprovider", imageFile)
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri)
                startActivityForResult(cameraIntent, cameraRequestCode)

            } else {
                Toast.makeText(applicationContext, getString(R.string.permission_camera), Toast.LENGTH_SHORT).show()
            }
        }

        val requestGalleryPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {

                val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                startActivityForResult(galleryIntent, galleryRequestCode)

            } else {
                Toast.makeText(applicationContext, getString(R.string.permission_gallery), Toast.LENGTH_SHORT).show()
            }
        }

        cameraButton.setOnClickListener {
            requestCameraPermission.launch(android.Manifest.permission.CAMERA)
        }

        galleryButton.setOnClickListener {
            requestGalleryPermission.launch(android.Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun predictAge (mat: Mat?): String {

        val model = BestModel.newInstance(applicationContext)
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 200, 200, 1), DataType.FLOAT32)
        val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * 200 * 200)
        byteBuffer.order(ByteOrder.nativeOrder())

        val floatValues = FloatArray(200 * 200)

        for (i in 0 until 200) {
            for (j in 0 until 200) {
                floatValues[i * 200 + j] = mat!!.get(i, j)[0].toFloat()
            }
        }

        for (value in floatValues) {
            byteBuffer.putFloat(value)
        }

        inputFeature0.loadBuffer(byteBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val output = outputFeature0.floatArray[0].roundToInt().toString()

        model.close()
        return output
    }

    private fun rotateBitmap(bitmap: Bitmap?, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
        }
        return Bitmap.createBitmap(bitmap!!, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun loadRotatedBitmapFromGallery(imageUri: Uri): Bitmap {
        val inputStream = contentResolver.openInputStream(imageUri)
        var bitmap = BitmapFactory.decodeStream(inputStream)
        val exif = ExifInterface(contentResolver.openInputStream(imageUri)!!)
        val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        bitmap = rotateBitmap(bitmap, orientation)
        inputStream?.close()
        return bitmap
    }

    private fun loadRotatedBitmapFromCamera(): Bitmap{
        val tempImage = File(currentPhotoPath)
        val exif = ExifInterface(tempImage.absolutePath)
        val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED)
        var bitmap = BitmapFactory.decodeFile(currentPhotoPath)
        bitmap = rotateBitmap(bitmap, orientation)
        tempImage.delete()
        return bitmap
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {

        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {

            var imageBitmap = null as Bitmap?

            if (requestCode == cameraRequestCode) {
                imageBitmap = loadRotatedBitmapFromCamera()
            }
            else if (requestCode == galleryRequestCode){
                val imageUri: Uri? = data?.data
                imageBitmap = loadRotatedBitmapFromGallery(imageUri!!)
            }

            val mat = Mat()
            Utils.bitmapToMat(imageBitmap, mat)
            val grayMat = Mat()
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)
            val faces = MatOfRect()

            val inputStream: InputStream = resources.openRawResource(R.raw.haarcascade_frontalface_default)
            val cascadeClassifier = CascadeClassifier()
            val file: File = File.createTempFile("temp", "xml")
            val outputStream: OutputStream = FileOutputStream(file)
            val buffer = ByteArray(inputStream.available())
            inputStream.read(buffer)
            outputStream.write(buffer)
            inputStream.close()
            outputStream.close()
            cascadeClassifier.load(file.absolutePath)
            cascadeClassifier.detectMultiScale(grayMat, faces, 1.1, 3, 0, Size(200.0, 200.0))

            val imageView = findViewById<ImageView>(R.id.image)
            val textView = findViewById<TextView>(R.id.text)

            if (!faces.empty()) {
                val face = faces.toList()[0]
                val croppedMat = mat.submat(face)
                val croppedBitmap = Bitmap.createBitmap(face.width, face.height, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(croppedMat, croppedBitmap)
                imageView.setImageBitmap(croppedBitmap)
                textView.text = getString(R.string.face_detected)

                val croppedGrayMat = grayMat.submat(face)

                croppedGrayMat.convertTo(croppedGrayMat, CvType.CV_32F, 1.0 / 255.0)

                val resizedMat = Mat()
                Imgproc.resize(croppedGrayMat, resizedMat, Size(200.0, 200.0))

                val predictedAge = getString(R.string.predicted_age)+" "+predictAge(resizedMat)
                textView.text = predictedAge
            }

            else {
                imageView.setImageBitmap(imageBitmap)
                textView.text = getString(R.string.face_not_detected)
            }
        }
    }
}