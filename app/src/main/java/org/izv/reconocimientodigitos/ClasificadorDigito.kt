package org.izv.reconocimientodigitos

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks.call
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.Interpreter
import java.text.DecimalFormat

class ClasificadorDigito(private val context: Context) {

    private var interpreter: Interpreter? = null

    private val servicioEjecucionSegundoPlano: ExecutorService = Executors.newCachedThreadPool()

    private var alto: Int = 0
    private var ancho: Int = 0
    private var tamanoModelo: Int = 0

    var isInicializado = false
        private set

    fun clasifica(bitmap: Bitmap): Task<String> {
        return call(servicioEjecucionSegundoPlano, Callable<String> { clasificaAsincrono(bitmap) })
    }

    private fun clasificaAsincrono(bitmap: Bitmap): String {
        check(isInicializado) { "TF Lite no está inicializado" }

        // Pre-processing: resize the input image to match with model input shape
        val imagenRedimensionada = Bitmap.createScaledBitmap(bitmap, ancho, alto, true)
        val buffer = convierteBitmapAByteBuffer(imagenRedimensionada)
        // Define an array to store the model output
        val salida = Array(1) { FloatArray(NUMERO_DE_DIGITOS) }
        // Run inference with the input data
        interpreter?.run(buffer, salida)
        // Post-processing: find the digit that has highest probability and return it a human-readable string
        val resultado = salida[0]
        //result es una matriz de 10 elementos (para los dígitos del 0 al 9)
        //el elemento que tenga el valor más alto es el que se infiere
        for(indice in resultado) {
            Log.v(TAG, "confianza: " + (indice * 100))
        }
        val maxIndice = resultado.indices.maxBy { resultado[it] } ?: -1
        val formatoDecimal = DecimalFormat("#.##")
        val cadenaResultado = "Predicción: " + maxIndice + ".\n Confianza: " + formatoDecimal.format(resultado[maxIndice] * 100) + "%."
        return cadenaResultado
    }

    fun cierra() {
        call(
            servicioEjecucionSegundoPlano,
            Callable<String> {
                interpreter?.close()
                Log.d(TAG, "Closed TFLite interpreter.")
                null
            }
        )
    }

    private fun convierteBitmapAByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(tamanoModelo)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(ancho * alto)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (valorPixel in pixels) {
            val r = (valorPixel shr 16 and 0xFF)
            val g = (valorPixel shr 8 and 0xFF)
            val b = (valorPixel and 0xFF)
            // convert RGB to grayscale and normalize pixel value to [0..1]
            val valorNormalizado = (r + g + b) / 3.0f / 255.0f
            byteBuffer.putFloat(valorNormalizado)
        }
        return byteBuffer
    }

    fun inicializa(): Task<Void> {
        return call(
            servicioEjecucionSegundoPlano,
            Callable<Void> {
                inicializaAsincrono()
                null
            }
        )
    }

    @Throws(IOException::class)
    private fun inicializaAsincrono() {
        // Load the TF Lite model from asset folder
        val gestorAssets = context.assets
        val modelo = cargarArchivoModelo(gestorAssets, "mnist.tflite")
        // Initialize TF Lite Interpreter with NNAPI enabled
        val opciones = Interpreter.Options()
        opciones.setUseNNAPI(true)
        val interpreter = Interpreter(modelo, opciones)
        // Read input shape from model file
        val figuraEntrada = interpreter.getInputTensor(0).shape()
        ancho = figuraEntrada[1]
        alto = figuraEntrada[2]
        tamanoModelo = BYTES_POR_FLOAT * ancho * alto * PIXEL
        this.interpreter = interpreter
        isInicializado = true
    }

    @Throws(IOException::class)
    private fun cargarArchivoModelo(assetManager: AssetManager, filename: String): ByteBuffer {
        val descriptorArchivo = assetManager.openFd(filename)
        val flujoEntrada = FileInputStream(descriptorArchivo.fileDescriptor)
        val canalArchivo = flujoEntrada.channel
        val desplazamientoInicial = descriptorArchivo.startOffset
        val longitudDeclarada = descriptorArchivo.declaredLength
        return canalArchivo.map(FileChannel.MapMode.READ_ONLY, desplazamientoInicial, longitudDeclarada)
    }

    companion object {
        private const val BYTES_POR_FLOAT = 4
        private const val NUMERO_DE_DIGITOS = 10
        private const val PIXEL = 1
        private const val TAG = "ClasificadorDigito"
    }
}