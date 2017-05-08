
// check input parameters
void dictLearningOnline(double * X, int n_features, int n_samples, 
	int n_components, double alpha){
	init_dict = //
	
	// initialize dictionary => set values to zero, calloc

	// n_components <= rows => select first n_components rows
	// else padding with zeros at bottom
	// shuffle => not needed



	// generate batches
	// gen_batches()

	//initialize A and B to zeros
	int i;
	double * A = ;
	double * B = ;
	int max_iter = 1000;
	int nSamples;
	int nComponents;
	double * xt = (double*)malloc(*sizeof(double*));
	for(i = 0; i < max_iter; i++){
		// draw data sequentially
		int startIdx = ;
		int endIdx = ;
		
		xt = 
		// update dictionary
		// update parameters
		double regularization = ;
		double lambda = regularization/n_features;
		solveLars(xt, Dt, lambda, alphat);
		//update At;
		//update Bt;
		updateDictionary(Dt, At, Bt, ...);

		/*	// additional criterion for stopping the iteration
			if (l2norm(alphaOld, alphaNew)<stopCriterion){
				break;
			}
		*/


	}


	//sparse_codint = lars_solution


}

/*
	n_components = none => number of cols of X
*/