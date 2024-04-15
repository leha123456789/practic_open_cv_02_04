#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp> 
#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
using namespace cv;
using namespace std;
unordered_set<string> usedCards;




int main() 
{
    setlocale(LC_ALL, "Russian");
    vector<Mat> cardsImages;
    vector<string> cardsNames;
    vector<Mat> cardsDescriptors;
    vector<vector<KeyPoint>> cardsKeypoints;
    vector<vector<KeyPoint>> extractedCardsKeypoints;
    vector<Mat> extractedCardsDescriptors;
    Mat image = imread("карты.jpeg");
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);
    Mat edges;
    Mat blurred;
    GaussianBlur(gray, blurred, Size(3, 3), 0, 0);
    Laplacian(gray, edges, CV_8U, 5);
    threshold(edges, edges, 50, 255, THRESH_BINARY);
    Mat thresholded;
    adaptiveThreshold(edges, thresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
    Canny(gray, edges, 50, 150);
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat contoursImage = image.clone(); 
    drawContours(contoursImage, contours, -1, Scalar(0, 255, 0), 2);
    imshow("Contours", contoursImage);
    Mat mask = Mat::zeros(image.size(), CV_8U);
    for (const auto& contour : contours) 
    {
        fillPoly(mask, vector<vector<Point>>{contour}, Scalar(255));
        Mat card;
        image.copyTo(card, mask);
        RotatedRect rotatedRect = minAreaRect(contour);
        float angle = rotatedRect.angle;
        Point2f center = rotatedRect.center;
        Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
        Mat rotatedCard;
        warpAffine(card, rotatedCard, rotationMatrix, card.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar());
        resize(rotatedCard, card, Size(145, 220));
        Ptr<Feature2D> sift = SIFT::create();
        vector<KeyPoint> keypoints;
        Mat descriptors;
        sift->detectAndCompute(rotatedCard, noArray(), keypoints, descriptors);
        extractedCardsKeypoints.push_back(keypoints);
        extractedCardsDescriptors.push_back(descriptors);
        waitKey(0);
    }
    Mat card;
    card = imread("dama.jpeg");
    resize(card, card, Size(165, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("dama_bubi");

    card = imread("desatka.jpeg");
    resize(card, card, Size(165, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("desatka_vini");

    card = imread("devatka.jpeg");
    resize(card, card, Size(160, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("devatka_vini");

    card = imread("korol.jpeg");
    resize(card, card, Size(160, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("korol_bubi");

    card = imread("semerka.jpg");
    resize(card, card, Size(160, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("semerka_piki");

    card = imread("tuz.jpg");
    resize(card, card, Size(160, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("tuz_bubi");

    card = imread("tuzpiki.jpg");
    resize(card, card, Size(160, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("tuz_piki");

    card = imread("valet.jpg");
    resize(card, card, Size(160, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("valet_kresti");

    card = imread("vosmerka.jpg");
    resize(card, card, Size(160, 230));
    cardsImages.push_back(card);
    cardsNames.push_back("vosmerka_chervi");

    Ptr<Feature2D> sift = SIFT::create();
    for (int i = 0; i < cardsImages.size(); i++)
    {
        Mat dis;
        vector<KeyPoint> keypoints;
        sift->detectAndCompute(cardsImages[i], noArray(), keypoints, dis);
        cardsKeypoints.push_back(keypoints);
        cardsDescriptors.push_back(dis);
    }

    cout << "Количество загруженных карт: " << cardsImages.size() << endl;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    cout << "Размер scannedCardsDescriptors: " << extractedCardsDescriptors.size() << endl;
    cout << "Размер cardsDescriptors: " << cardsDescriptors.size() << endl;
    cout << "Размер scannedCardsKeypoints: " << extractedCardsKeypoints.size() << endl;
    cout << "Размер cardsKeypoints: " << cardsKeypoints.size() << endl;
    cout << "Размер cardsKeypoints: " << cardsNames.size() << endl;
    unordered_set<string> usedMostCommonCards;
    Mat outputImage = image.clone(); 
    for (size_t i = 0; i < extractedCardsDescriptors.size(); ++i) 
    {
        if (extractedCardsDescriptors[i].empty()) 
        {
            continue;
        }
        vector<int> matchedIndexes;
        vector<DMatch> matches;
        const double MAX_DISTANCE_THRESHOLD = 100.0;
        matcher->match(extractedCardsDescriptors[i], cardsDescriptors, matches);
        sort(matches.begin(), matches.end());
        for (const auto& match : matches)
        {
            if (match.distance < MAX_DISTANCE_THRESHOLD) 
            {
                matchedIndexes.push_back(match.trainIdx);
            }
        }
        if (!matchedIndexes.empty()) 
        {
            unordered_map<string, int> countMap;
            for (int idx : matchedIndexes)
            {
                if (idx >= 0 && idx < cardsNames.size()) 
                {
                    countMap[cardsNames[idx]]++;
                }
            }
            string mostCommonCard;
            int maxCount = 0;
            for (const auto& pair : countMap)
            {
                if (pair.second > maxCount&& usedMostCommonCards.find(pair.first) == usedMostCommonCards.end()) 
                {
                    maxCount = pair.second;
                    mostCommonCard = pair.first;
                }
            }
            cout << "Наиболее часто совпадающая карта для извлеченной карты " << i << ": " << mostCommonCard << endl;
            if (!mostCommonCard.empty()) 
            {
                vector<vector<Point>> contours_poly(1);
                approxPolyDP(contours[i], contours_poly[0], 3, true);
                Rect boundRect = boundingRect(contours_poly[0]);
                Point textOrg(boundRect.x + 10, boundRect.y + boundRect.height - 20);
                putText(image, mostCommonCard, textOrg, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                usedMostCommonCards.insert(mostCommonCard);
            }
        }
    }
    imshow("Карты с названиями", image);
    waitKey(0);
    return 0;
}
