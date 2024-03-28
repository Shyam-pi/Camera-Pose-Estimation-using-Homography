import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import warnings

warnings.filterwarnings("ignore")

def getTrack(vid_path : str, disp_vid = 'True', disp_plot = 'True'):
    cap = cv2.VideoCapture(vid_path)
 
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Hough search space
    beta = np.linspace(-np.pi/2,np.pi/2,180)
    rho_max = int(np.sqrt(1920**2 + 1080**2))
    rho = np.linspace(0,rho_max,800)
    lines = []

    for i in rho:
        for j in beta:
            lines.append([i,j])
        
    lines = np.array(lines)
 
    x_range = np.linspace(0,1920,10)

    num = 1
    final_corners = []
    
    # Read until video is completed
    while(cap.isOpened()):
        print(f"Frame processed = {num} / 148", end="\r", flush=True)
        num+=1

        # Capture frame-by-frame
        ret, og_frame = cap.read()
        count = np.zeros(lines.shape[0])

        if ret == True:
            
            frame = cv2.GaussianBlur(og_frame,(5,5),cv2.BORDER_DEFAULT)
            
            edges = cv2.Canny(frame,150,350)
            pts = np.argwhere(edges != 0)

            for pt in pts:
                res = np.round((pt[1]*np.cos(lines[:,1]) + pt[0]*np.sin(lines[:,1])) - lines[:,0]).astype(np.int32)
                count[res == 0] += 1
            
            top = np.sort(count.flatten())[-20:][::-1]
            
            ind = np.argwhere(count == top[0])[0]
            hypothesis = lines[ind,:][0]
            best = [hypothesis]

            for t in top:
                ind = np.argwhere(count == t)[0]
                hypothesis = lines[ind,:][0]
                
                # Perform duplicate check

                temp = np.array(best)
                # print(f"dist = {np.abs(temp[:,0] - hypothesis[0])}")
                if np.argwhere(np.abs(temp[:,0] - hypothesis[0]) < 140).size == 0:
                    best.append(hypothesis)
                # print(f"temp = {np.array(best)}")
                
                if len(best) == 4:
                    break
            
            best = np.array(best)
            
            # Finding corner points
            corners = set()
            for line1 in best:
                for line2 in best:
                    if np.array_equal(line1,line2):
                        continue
                    x = (line1[0]/np.sin(line1[1]) - line2[0]/np.sin(line2[1]))*(1/(1/np.tan(line1[1]) - 1/np.tan(line2[1])))
                    y = (line1[0] - x*np.cos(line1[1]))/np.sin(line1[1])
                    if not(0<x<1920) or not(0<y<1080):
                        continue
                    corners.add((y.round(3),x.round(3)))
                    # corners.append([y,x])

            corners = list(corners)
            corners = np.array(corners)

            if corners.shape[0] == 4:
                lc = corners[np.argwhere(corners[:,1] == np.min(corners[:,1])), :]
                rc = corners[np.argwhere(corners[:,1] == np.max(corners[:,1])), :]
                tc = corners[np.argwhere(corners[:,0] == np.min(corners[:,0])), :]
                bc = corners[np.argwhere(corners[:,0] == np.max(corners[:,0])), :]

                final_corners.append(np.array([lc,tc,rc,bc]))

            # Plotting
            if disp_plot:
                fig, ax = plt.subplots()
                im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for line in best:
                    ax.plot(x_range, (line[0] - x_range*np.cos(line[1]))/(np.sin(line[1])), color='red')
                ax.scatter(corners[:,1],corners[:,0], s=5, color='red')
                ax.set_xlabel('X coordinates')
                ax.set_ylabel('Y coordinates')
                ax.set_xlim(0, 1920)
                ax.set_ylim(0, 1080)
                ax.invert_yaxis()
                
                plt.show(block=False)
                plt.pause(1)
                plt.close()
                # plt.show()

            # Press Q on keyboard to  exit
            if disp_vid:
                cv2.imshow('Edges', edges)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
    
    # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

    return final_corners

def getH(final_corners, K):
    R_bank = []
    T_bank = []

    # Detected points in world coordinate system
    P = np.array([[0,0,1],[0,0.279,1],[0.216,0.279,1],[0.216,0,1]])

    for i in range(len(final_corners)):
        p = final_corners[i].reshape((4,2))
        A = np.zeros((p.shape[0]*2,9))

        for i in range(p.shape[0]):
            x = P[i,0]
            x_ = p[i,1]
            y = P[i,1]
            y_ = p[i,0]
            A[i*2:i*2 + 2,:] = np.array([[-x, -y, -1, 0, 0, 0, x*x_, y*x_, x_],
                                        [0, 0, 0, -x, -y, -1, x*y_, y*y_, y_]])
            
        # SVD solution
        _ , _ , V = np.linalg.svd(A)
        h = V.T[:,8]
        H = np.reshape(h, (3,3))

        E = np.linalg.inv(K) @ H

        x_unit = E[:,0]/np.linalg.norm(E[:,0])
        y_unit = E[:,1]/np.linalg.norm(E[:,1])
        scale = 2/(np.linalg.norm(E[:,0]) + np.linalg.norm(E[:,1]))
        T = E[:,-1] * scale

        z_unit = np.cross(x_unit,y_unit)

        R = np.vstack((x_unit, y_unit, z_unit)).T

        r =  Rotation.from_matrix(R)
        angles = r.as_euler("xyz",degrees=True)

        R_bank.append(angles)
        T_bank.append(T)
    
    return R_bank, T_bank

def main():
    vid_path = "project2.avi"
    f_corners = getTrack(vid_path=vid_path, disp_vid=False, disp_plot=False)
    K = np.array([[1380,0,946],[0,1380,527],[0,0,1]])

    R_bank, T_bank = getH(final_corners=f_corners, K = K)
    R_bank = np.array(R_bank)
    T_bank = np.array(T_bank)

    """Plotting the translation and rotation of the camera's coordinate frame against the frame number"""

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    x = range(T_bank.shape[0])

    # Plot the first graph in the first subplot
    y1 = R_bank[:,0]
    y2 = R_bank[:,1]
    y3 = R_bank[:,2]

    # Plot the three graphs in the same plot
    axs[0].plot(x, y1, label='Roll')
    axs[0].plot(x, y2, label='Pitch')
    axs[0].plot(x, y3, label='Yaw')

    # Add labels and title
    axs[0].set_xlabel('Frame #')
    axs[0].set_ylabel('Value in degrees')
    axs[0].set_title('Rotation change in the camera coordinates')
    axs[0].legend()

    # Plot the second graph in the second subplot
    y1 = T_bank[:,0]
    y2 = T_bank[:,1]
    y3 = T_bank[:,2]

    # Plot the three graphs in the same plot
    axs[1].plot(x, y1, label='T_x')
    axs[1].plot(x, y2, label='T_y')
    axs[1].plot(x, y3, label='T_z')

    # Add labels and title
    axs[1].set_xlabel('Frame #')
    axs[1].set_ylabel('Value in metres')
    axs[1].set_title('Translation change in the camera coordinates')

    axs[1].legend()

    # Add labels and a title for the entire figure
    fig.suptitle('Camera position and rotation changes')
    plt.show()
    fig.savefig('results/hough.png', dpi=300)

if __name__ == "__main__":
    main()