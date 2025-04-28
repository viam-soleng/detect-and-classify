package main

import (
	"context"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/services/vision"
	"go.viam.com/utils"

	visionsvc "github.com/viam-soleng/detect-and-classify/visionsvc"
)

func main() {
	utils.ContextualMain(mainWithArgs, module.NewLoggerFromArgs("Module: Detect and Classify"))
}

func mainWithArgs(ctx context.Context, args []string, _ logging.Logger) (err error) {
	// instantiates the module itself
	myMod, err := module.NewModuleFromArgs(ctx)
	if err != nil {
		return err
	}

	// Models and APIs add helpers to the registry during their init().
	// They can then be added to the module here.
	err = myMod.AddModelFromRegistry(ctx, vision.API, visionsvc.Model)
	if err != nil {
		return err
	}

	// Each module runs as its own process
	err = myMod.Start(ctx)
	defer myMod.Close(ctx)
	if err != nil {
		return err
	}
	<-ctx.Done()
	return nil
}
