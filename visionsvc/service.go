package visionsvc

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"os"
	"slices"
	"sort"
	"sync"

	"github.com/pkg/errors"
	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/vision"
	vis "go.viam.com/rdk/vision"
	"go.viam.com/rdk/vision/classification"
	"go.viam.com/rdk/vision/objectdetection"
	"go.viam.com/rdk/vision/viscapture"
)

var errUnimplemented = errors.New("unimplemented")
var Model = resource.NewModel("viam-soleng", "vision", "detect-and-classify")
var PrettyName = "Viam detect and classify vision service"
var Description = "A module of the Viam vision service that crops an image to an initial detection bounding box and then processes the cropped image with the provided vision service"

type Config struct {
	Camera              string   `json:"camera"`
	Detector            string   `json:"detector_service"`
	DetectorConfidence  float64  `json:"detector_confidence"`
	MaxDetections       int      `json:"max_detections"`
	DetectorValidLabels []string `json:"detector_valid_labels"`
	DetBorder           int      `json:"padding"`
	VisionService       string   `json:"vision_service"`
	LogImage            bool     `json:"log_images"`
	ImagePath           string   `json:"images_path"`
}

type myVisionSvc struct {
	resource.Named
	logger              logging.Logger
	camera              camera.Camera
	detector            vision.Service
	detectorConfidence  float64
	maxDetections       int
	detectorValidLabels []string
	detBorder           int
	visionService       vision.Service
	logImage            bool
	imagePath           string
	mu                  sync.RWMutex
	cancelCtx           context.Context
	cancelFunc          func()
	done                chan bool
}

func init() {
	resource.RegisterService(
		vision.API,
		Model,
		resource.Registration[vision.Service, *Config]{
			Constructor: newService,
		})
}

func newService(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (vision.Service, error) {
	logger.Debugf("Starting %s %s", PrettyName)
	cancelCtx, cancelFunc := context.WithCancel(context.Background())

	svc := myVisionSvc{
		Named:      conf.ResourceName().AsNamed(),
		logger:     logger,
		cancelCtx:  cancelCtx,
		cancelFunc: cancelFunc,
		mu:         sync.RWMutex{},
		done:       make(chan bool),
	}

	if err := svc.Reconfigure(ctx, deps, conf); err != nil {
		return nil, err
	}
	return &svc, nil
}

func (cfg *Config) Validate(path string) ([]string, error) {
	if cfg.Camera == "" {
		return nil, errors.New(`"camera" is required`)
	}
	if cfg.Detector == "" {
		return nil, errors.New(`"detector_service" is required`)
	}
	if cfg.DetectorConfidence <= 0.0 {
		return nil, errors.New(`"detector_confidence" must be >= 0.0`)
	}
	if cfg.VisionService == "" {
		return nil, errors.New(`"vision_service" is required`)
	}
	return []string{cfg.Camera, cfg.Detector, cfg.VisionService}, nil
}

// Reconfigure reconfigures with new settings.
func (svc *myVisionSvc) Reconfigure(ctx context.Context, deps resource.Dependencies, conf resource.Config) error {
	svc.mu.Lock()
	defer svc.mu.Unlock()
	svc.logger.Debugf("Reconfiguring %s", PrettyName)
	// In case the module has changed name
	svc.Named = conf.ResourceName().AsNamed()
	newConf, err := resource.NativeConfig[*Config](conf)
	if err != nil {
		return err
	}
	// Get the camera
	svc.camera, err = camera.FromDependencies(deps, newConf.Camera)
	if err != nil {
		return errors.Wrapf(err, `unable to get the "camera": %v for image sourcing...`, newConf.Detector)
	}
	// Get the face cropper
	svc.detector, err = vision.FromDependencies(deps, newConf.Detector)
	if err != nil {
		return errors.Wrapf(err, `unable to get "detector_service": %v for image cropping...`, newConf.Detector)
	}
	// Get the detector confidence threshold
	svc.detectorConfidence = newConf.DetectorConfidence
	// Get the detector dependency
	svc.visionService, err = vision.FromDependencies(deps, newConf.VisionService)
	if err != nil {
		return errors.Wrapf(err, `unable to get "vision_service": %v `, newConf.VisionService)
	}
	svc.detBorder = newConf.DetBorder
	svc.maxDetections = newConf.MaxDetections
	svc.detectorValidLabels = newConf.DetectorValidLabels
	svc.logImage = newConf.LogImage
	svc.imagePath = newConf.ImagePath
	svc.logger.Debug("**** Reconfigured ****")
	return nil
}

// Classifications can be implemented to extend functionality but returns unimplemented currently.
func (svc *myVisionSvc) Classifications(ctx context.Context, img image.Image, n int, extra map[string]interface{}) (classification.Classifications, error) {
	croppedImages, err := svc.cropDetections(ctx, img)
	if err != nil {
		return nil, err
	}
	result, err := svc.classify(ctx, croppedImages, n)
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ClassificationsFromCamera can be implemented to extend functionality but returns unimplemented currently.
func (svc *myVisionSvc) ClassificationsFromCamera(ctx context.Context, cameraName string, n int, _ map[string]interface{}) (classification.Classifications, error) {
	image, err := camera.DecodeImageFromCamera(ctx, "", nil, svc.camera)
	if err != nil {
		return nil, err
	}
	images, err := svc.cropDetections(ctx, image)
	if err != nil {
		return nil, err
	}
	return svc.classify(ctx, images, n)
}

func (svc *myVisionSvc) Detections(ctx context.Context, image image.Image, extra map[string]interface{}) ([]objectdetection.Detection, error) {
	images, err := svc.cropDetections(ctx, image)
	if err != nil {
		return nil, err
	}
	return svc.detect(ctx, images)
}

func (svc *myVisionSvc) DetectionsFromCamera(ctx context.Context, cameraName string, _ map[string]interface{}) ([]objectdetection.Detection, error) {
	image, err := camera.DecodeImageFromCamera(ctx, "", nil, svc.camera)
	if err != nil {
		return nil, err
	}
	images, err := svc.cropDetections(ctx, image)
	if err != nil {
		return nil, err
	}
	return svc.detect(ctx, images)
}

// ObjectPointClouds can be implemented to extend functionality but returns unimplemented currently.
func (s *myVisionSvc) GetObjectPointClouds(ctx context.Context, cameraName string, extra map[string]interface{}) ([]*vis.Object, error) {
	return nil, errUnimplemented
}

// CaptureAllFromCamera can be implemented to extend functionality but returns unimplemented currently.
func (vm *myVisionSvc) CaptureAllFromCamera(
	ctx context.Context,
	cameraName string,
	opt viscapture.CaptureOptions,
	extra map[string]interface{},
) (viscapture.VisCapture, error) {
	return viscapture.VisCapture{}, errUnimplemented
}

// GetProperties can be implemented to extend functionality but returns unimplemented currently.
func (vm *myVisionSvc) GetProperties(context.Context, map[string]interface{}) (*vision.Properties, error) {
	return nil, errUnimplemented
}

// DoCommand can be implemented to extend functionality but returns unimplemented currently.
func (s *myVisionSvc) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return nil, errUnimplemented
}

// The close method is executed when the component is shut down
func (svc *myVisionSvc) Close(ctx context.Context) error {
	svc.logger.Debugf("Shutting down %s", PrettyName)
	svc.camera.Close(ctx)
	return nil
}

// Take an input image, detect objects, crop the image down to the detected bounding box and
// hand over to classifier for more accurate classifications
func (svc *myVisionSvc) cropDetections(ctx context.Context, img image.Image) ([]image.Image, error) {
	// Get detections from the provided Image
	detections, err := svc.detector.Detections(ctx, img, nil)
	if err != nil {
		return nil, err
	}
	// Filter detections by detector confidence level and valid labels settings
	filterFunc := func(detection objectdetection.Detection) bool {
		return (detection.Score() >= svc.detectorConfidence) && (slices.Contains(svc.detectorValidLabels, detection.Label()) || len(svc.detectorValidLabels) == 0)
	}
	detections = filter(detections, filterFunc)

	// Sort filtered detections based upon score
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Score() > detections[j].Score()
	})
	// Trim detections based upon max detections setting / if detectorMaxDetections = 0 -> no limit
	if len(detections) > svc.maxDetections && svc.maxDetections != 0 {
		detections = detections[:svc.maxDetections]
	}
	svc.logger.Debugf("Detections #: %v/%v", len(detections), svc.maxDetections)
	svc.logger.Debugf("Detections Details: %v", detections)
	croppedImages := []image.Image{}
	for _, detection := range detections {
		// Increase/decrease bounding box according to detection border setting
		rectangle := image.Rect(
			detection.BoundingBox().Min.X-svc.detBorder,
			detection.BoundingBox().Min.Y-svc.detBorder,
			detection.BoundingBox().Max.X+svc.detBorder,
			detection.BoundingBox().Max.Y+svc.detBorder)
		croppedImg, err := cropImage(img, &rectangle)
		if err != nil {
			return nil, err
		}
		// Save cropped images to disk
		if svc.logImage {
			err := saveImage(croppedImg, svc.imagePath)
			if err != nil {
				return nil, err
			}
		}
		croppedImages = append(croppedImages, croppedImg)
	}
	return croppedImages, nil
}

// Pass the cropped images to the configured classification vision service and get the classifications with the highest confidence
func (svc *myVisionSvc) classify(ctx context.Context, images []image.Image, n int) (classification.Classifications, error) {
	classificationResult := classification.Classifications{}
	for _, image := range images {
		class, err := svc.visionService.Classifications(ctx, image, n, nil)
		if err != nil {
			return nil, err
		}
		classificationResult = append(classificationResult, class...)
	}
	sort.Slice(classificationResult, func(i, j int) bool {
		return classificationResult[i].Score() > classificationResult[j].Score()
	})
	return classificationResult, nil
}

// Pass the cropped images to the configured detection vision service and get the detections with the highest confidence
func (svc *myVisionSvc) detect(ctx context.Context, images []image.Image) ([]objectdetection.Detection, error) {
	detectionResult := []objectdetection.Detection{}
	for _, image := range images {
		class, err := svc.visionService.Detections(ctx, image, nil)
		if err != nil {
			return nil, err
		}
		detectionResult = append(detectionResult, class...)
	}
	sort.Slice(detectionResult, func(i, j int) bool {
		return detectionResult[i].Score() > detectionResult[j].Score()
	})
	if len(detectionResult) > svc.maxDetections && svc.maxDetections != 0 {
		detectionResult = detectionResult[:svc.maxDetections]
	}
	return detectionResult, nil
}

// Crops images based upon bounding box rectangles
func cropImage(img image.Image, rect *image.Rectangle) (image.Image, error) {
	// The cropping operation is done by creating a new image of the size of the rectangle
	// and drawing the relevant part of the original image onto the new image.
	cropped := image.NewRGBA(rect.Bounds())
	draw.Draw(cropped, rect.Bounds(), img, rect.Min, draw.Src)
	return cropped, nil
}

// Saves images to a path on disk
func saveImage(image image.Image, imagePath string) error {
	buf := new(bytes.Buffer)
	err := jpeg.Encode(buf, image, nil)
	if err != nil {
		return err
	}
	digest := sha256.New()
	digest.Write(buf.Bytes())
	hash := digest.Sum(nil)
	f, err := os.Create(fmt.Sprintf("%v/%x.jpg", imagePath, hash))
	if err != nil {
		return err
	}
	defer f.Close()
	opt := jpeg.Options{
		Quality: 90,
	}
	jpeg.Encode(f, image, &opt)
	return nil
}

// Generic helper function to filter slices
func filter[T any](ss []T, test func(T) bool) (ret []T) {
	for _, s := range ss {
		if test(s) {
			ret = append(ret, s)
		}
	}
	return
}
