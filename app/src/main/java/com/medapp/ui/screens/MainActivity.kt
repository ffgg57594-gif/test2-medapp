package com.medapp.ui.screens

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.medapp.databinding.ActivityMainBinding

/**
 * Landing screen. Its only job is to route to the model picker.
 * Kept intentionally minimal — this is a fixed entry point that won't need
 * to change as models are added.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnBrowseModels.setOnClickListener {
            startActivity(Intent(this, ModelListActivity::class.java))
        }
    }
}
