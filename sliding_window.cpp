/*
* Sliding Window Implementation (Works better than Selective Search)
*/
// Parameters of your slideing window
// int windows_n_rows = 120;
// int windows_n_cols = 120;

// // Step of each window
// int StepSlide = 5;

// // Cycle row step
// for (int row = 0; row <= image.rows - windows_n_rows; row += StepSlide)
// {
//     // Cycle col step
//     for (int col = 0; col <= image.cols - windows_n_cols; col += StepSlide)
//     {
//         cv::Rect window(col, row, windows_n_rows, windows_n_cols);
//         cv::Mat image_proposal = image(window);

//         hog.computeHOG(image_proposal, descriptors);
        
//         printf("Predicting label per region ...\n");
//         classifier.predict_one(cv::Mat1f(descriptors), proposal_label, proposal_confidence);

//         // Harsh Threshold On Confidence
//         // if(proposal_confidence < 0.85) {
//         //     proposal_label = 3;
//         // }

//         // Box Proposals Per Label
//         box_proposals_per_label[proposal_label].push_back(window);

//         if (proposal_label == 0 || proposal_label == 1 || proposal_label == 2) {
//             box_proposals_valid.push_back(window);
//         }

//         if (proposal_label == 0) {
//             cv::rectangle(
//                 visualization_image, 
//                 window, 
//                 cv::Scalar(255, 0, 0)
//             );
//         }
        
//         if (proposal_label == 1) {
//             cv::rectangle(
//                 visualization_image, 
//                 window, 
//                 cv::Scalar(0, 255, 0)
//             );
//         } 
        
//         if (proposal_label == 2) {
//             cv::rectangle(
//                 visualization_image, 
//                 window, 
//                 cv::Scalar(0, 0, 255)
//             );
//         }
//     }
// }