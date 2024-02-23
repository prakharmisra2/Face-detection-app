package com.example.newfacedetector;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.newfacedetector.ml.LiteModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    Button selectImageButton;
    ImageView dispalyImage;
    Button predictButton;
    TextView resultText;
    Bitmap bitmap;
    int SELECT_IMAGE = 200;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        selectImageButton = findViewById(R.id.selectImageButton);
        dispalyImage = findViewById(R.id.imageView);
        predictButton = findViewById(R.id.pridictButton);
        resultText = findViewById(R.id.resultText);

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new NormalizeOp(0.0f,48.0f))
                .add(new ResizeOp(48,48,ResizeOp.ResizeMethod.BILINEAR))
                .add(new TransformToGrayscaleOp())
                .build();


        selectImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                imageChooser();
            }
        });
        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {
                    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(bitmap);

                    tensorImage = imageProcessor.process(tensorImage);
                    LiteModel model = LiteModel.newInstance(getBaseContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 48, 48, 1}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(tensorImage.getBuffer());

                    // Runs model inference and gets result.
                    LiteModel.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    float [] output = outputFeature0.getFloatArray();
                    int max = 0;
                    for(int i = 0; i<output.length; i++){
                        if(output[max] < output[i]) max = i;
                    }
                    Map<Integer, String> labelsDict = new HashMap<>();
                   // emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
                    labelsDict.put(0, "Angry");
                    labelsDict.put(1, "Disgust");
                    labelsDict.put(2, "Fear");
                    labelsDict.put(3, "Happy");
                    labelsDict.put(4, "Neutral");
                    labelsDict.put(5, "Sad");
                    labelsDict.put(6, "Surprise");
                    String result = labelsDict.get(max);
                    resultText.setText(result);
                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });
    }
    public void imageChooser(){

        Intent i = new Intent();
        i.setType("image/*");
        i.setAction(Intent.ACTION_GET_CONTENT);

        startActivityForResult(Intent.createChooser(i, "Select Picture"), SELECT_IMAGE);
    }
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            // compare the resultCode with the
            // SELECT_PICTURE constant
            if (requestCode == SELECT_IMAGE) {
                // Get the url of the image f rom data
                Uri selectedImageUri = data.getData();

                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
                }catch (Exception e){
                    //nothing to do here.
                }
                if (null != selectedImageUri) {
                    // update the preview image in the layout
                    dispalyImage.setImageURI(selectedImageUri);
                }
            }
        }
    }

}