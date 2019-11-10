package org.izv.reconocimientodigitos

import android.annotation.SuppressLint
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.divyanshu.draw.widget.DrawView

class ActividadPrincipal : AppCompatActivity() {

    private var modeloML = ClasificadorDigito(this)

    private var botonLimpiar: Button? = null
    private var drawView: DrawView? = null
    private var textViewResultado: TextView? = null

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.actividad_principal)

        botonLimpiar = findViewById(R.id.btLimpiar)
        drawView = findViewById(R.id.dvDibujo)
        textViewResultado = findViewById(R.id.tvResultado)

        drawView?.setStrokeWidth(70.0f)
        drawView?.setColor(Color.WHITE)
        drawView?.setBackgroundColor(Color.BLACK)

        botonLimpiar?.setOnClickListener {
            drawView?.clearCanvas()
            textViewResultado?.text = getString(R.string.texto_resultado)
        }

        drawView?.setOnTouchListener { _, event ->
            drawView?.onTouchEvent(event)
            if (event.action == MotionEvent.ACTION_UP) {
                clasificaImagen()
            }
            true
        }

        modeloML.inicializa()
            .addOnFailureListener { e -> Log.e(TAG, "Error al inicializar", e) }
    }

    override fun onDestroy() {
        modeloML.cierra()
        super.onDestroy()
    }

    private fun clasificaImagen() {
        val bitmap = drawView?.getBitmap()

        if ((bitmap != null) && (modeloML.isInicializado)) {
            modeloML.clasifica(bitmap)
                .addOnSuccessListener { resultText ->
                    textViewResultado?.text = resultText
                }
                .addOnFailureListener { e ->
                    textViewResultado?.text = e.localizedMessage
                    Log.e(TAG, "Error al tratar de inferir.", e)
                }
        }
    }

    companion object {
        private const val TAG = "ActividadPrincipal"
    }
}